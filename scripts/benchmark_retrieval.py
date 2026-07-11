"""Standalone benchmark for Kenzo's Phase 5 "Semantic Memory and Retrieval
Engine" -- specifically, RETRIEVAL QUALITY, as a companion to
scripts/benchmark_indexing.py (Phase 4's INDEXING SPEED benchmark).

NOT a pytest test. It is a human-readable report generator: it builds a
realistic synthetic Java repo, indexes it with the real Phase 4 + Phase 5
pipeline, runs a fixed set of queries through both a "Before Phase 5"
baseline and the real "After Phase 5" `CodeSearchEngine`, and prints a
side-by-side comparison plus a final tally. It lives under scripts/, has no
test_* functions, and is never collected by the test suite.

"Before Phase 5" baseline
--------------------------
Phase 4 already ships `SymbolIndex.find_class`/`.find_function_or_method`
(case-insensitive EXACT name matching only) and the older
`kenzo.tools.symbols.find_symbol` tool (exact-then-substring over a flat
symbol list). Neither of those files is touched here. Instead, this script
implements its OWN inline baseline (`baseline_search` below) that generalizes
both into one honest, best-effort "purely structural/lexical Kenzo":

  (a) exact case-insensitive match on a symbol's simple or qualified name
  (b) else, substring match of the (lowercased) query against a symbol's
      simple or qualified name
  (c) else, plain keyword/token-overlap score of the query against each
      symbol's rendered SIGNATURE text (kind/stereotype/annotations/
      extends/implements/dependencies/endpoint/parameters/return type/
      method-name list -- structural metadata only, NEVER the symbol's
      Javadoc comment or source body, and NEVER a vector embedding or
      dependency-graph centrality boost)

This is deliberately a bit MORE capable than either real Phase 4 tool (it
adds the keyword-overlap fallback (c), which neither `find_class` nor
`find_symbol` has), specifically so the comparison is fair rather than a
strawman: it represents the best a purely-structural/lexical Kenzo could
plausibly do before Phase 5's hybrid engine existed, not the worst.

"After Phase 5" system under test
-----------------------------------
The real `kenzo.core.code_search.CodeSearchEngine`, built against a real
`SymbolIndex` + `ReferenceGraph` + `DependencyGraph` + `SemanticIndexer`
(default local `HashingEmbeddingProvider` -- offline, dependency-free,
lexical/fuzzy-substring similarity, NOT a trained semantic model; see
`kenzo/memory/vector/embeddings.py`'s module docstring) + `QdrantStore`
(embedded `:memory:` mode) -- exactly the wiring `IndexingService` uses in
production (see `kenzo/core/indexing_service.py`).

Fixture
-------
A 21-class synthetic Spring Boot-shaped Java repo spanning five concerns
(auth/logout, password reset, user management/permissions, billing/payments,
notifications/audit), with real Javadoc comments and real constructor-DI
call chains -- enough texture for conceptual/symptom queries to have
something real to find, not a toy 2-class repo.

Usage
-----
    C:\\Users\\sahil.r\\Ken\\.venv\\Scripts\\python.exe scripts\\benchmark_retrieval.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

# Allow running as `python scripts/benchmark_retrieval.py` from the repo root
# without requiring kenzo to already be on sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kenzo.config.settings import EmbeddingConfig  # noqa: E402
from kenzo.core.code_search import CodeSearchEngine, SearchResult  # noqa: E402
from kenzo.core.context_manager import _tokenize  # noqa: E402
from kenzo.intelligence.dependency_graph import DependencyGraph  # noqa: E402
from kenzo.intelligence.models import ClassSymbol, FunctionSymbol  # noqa: E402
from kenzo.intelligence.reference_graph import ReferenceGraph  # noqa: E402
from kenzo.intelligence.symbol_index import SymbolIndex  # noqa: E402
from kenzo.memory.vector.embeddings import EmbeddingRouter  # noqa: E402
from kenzo.memory.vector.semantic_indexer import SemanticIndexer  # noqa: E402
from kenzo.memory.vector.store import QdrantStore  # noqa: E402

# ---------------------------------------------------------------------------
# Fixture: a realistic Spring Boot-shaped Java repo, 21 classes across 5
# concerns (auth/logout, password reset, user management, billing/payments,
# notifications/audit) with real Javadoc comments and real constructor-DI
# call chains.
# ---------------------------------------------------------------------------

_FIXTURE_FILES: dict[str, str] = {
    "src/main/java/com/example/auth/controller/AuthController.java": """\
package com.example.auth.controller;

import com.example.auth.service.AuthService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * AuthController exposes the HTTP endpoints for signing users in and out of
 * the application, including refreshing an access token. Delegates all real
 * work to AuthService.
 */
@RestController
@RequestMapping("/api/auth")
public class AuthController {

    private final AuthService authService;

    public AuthController(AuthService authService) {
        this.authService = authService;
    }

    @PostMapping("/login")
    public String login(String username, String password) {
        return authService.authenticateUser(username, password);
    }

    /**
     * Signs the current user out: ends their session and revokes their
     * access token so the same credentials can't be reused after logout.
     */
    @PostMapping("/logout")
    public String logout(String sessionId) {
        return authService.logout(sessionId);
    }

    @PostMapping("/refresh")
    public String refreshToken(String refreshToken) {
        return authService.issueAccessToken(refreshToken);
    }

    @GetMapping("/health")
    public String health() {
        return "OK";
    }
}
""",
    "src/main/java/com/example/auth/controller/PasswordResetController.java": """\
package com.example.auth.controller;

import com.example.auth.service.PasswordResetService;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * PasswordResetController exposes endpoints for users who forgot their
 * password to request and complete a reset.
 */
@RestController
@RequestMapping("/api/auth/password")
public class PasswordResetController {

    private final PasswordResetService passwordResetService;

    public PasswordResetController(PasswordResetService passwordResetService) {
        this.passwordResetService = passwordResetService;
    }

    @PostMapping("/forgot")
    public String forgotPassword(String email) {
        return passwordResetService.initiatePasswordReset(email);
    }

    @PostMapping("/reset")
    public String resetPassword(String token, String newPassword) {
        return passwordResetService.completePasswordReset(token, newPassword);
    }
}
""",
    "src/main/java/com/example/auth/service/AuthService.java": """\
package com.example.auth.service;

import com.example.auth.repository.SessionRepository;
import com.example.auth.client.OidcClient;
import org.springframework.stereotype.Service;

/**
 * AuthService contains the core authentication business logic: verifying
 * credentials, issuing tokens, and tearing down a user's session (including
 * notifying the upstream identity provider) when they sign out.
 */
@Service
public class AuthService {

    private final SessionRepository sessionRepository;
    private final TokenService tokenService;
    private final OidcClient oidcClient;

    public AuthService(SessionRepository sessionRepository, TokenService tokenService, OidcClient oidcClient) {
        this.sessionRepository = sessionRepository;
        this.tokenService = tokenService;
        this.oidcClient = oidcClient;
    }

    public String authenticateUser(String username, String password) {
        sessionRepository.createSession(username);
        return tokenService.generateToken(username);
    }

    /**
     * Ends the user's session: invalidates the session record, revokes the
     * access token, and notifies the external identity provider so a single
     * logout signs the user out everywhere.
     */
    public String logout(String sessionId) {
        sessionRepository.invalidateSession(sessionId);
        tokenService.revokeToken(sessionId);
        oidcClient.notifyIdentityProviderOfLogout(sessionId);
        return "logged out";
    }

    public String issueAccessToken(String refreshToken) {
        return tokenService.generateToken(refreshToken);
    }
}
""",
    "src/main/java/com/example/auth/service/TokenService.java": """\
package com.example.auth.service;

import org.springframework.stereotype.Service;

/**
 * TokenService issues, validates, and revokes JWT access tokens.
 */
@Service
public class TokenService {

    public String generateToken(String userId) {
        return "token-for-" + userId;
    }

    public boolean validateToken(String token) {
        return token != null && !token.isEmpty();
    }

    /**
     * Revokes an access token so it can no longer be used, e.g. after the
     * user signs out or their session is otherwise terminated.
     */
    public void revokeToken(String token) {
        // mark token as revoked
    }
}
""",
    "src/main/java/com/example/auth/service/PasswordResetService.java": """\
package com.example.auth.service;

import com.example.user.repository.UserRepository;
import com.example.notification.service.NotificationService;
import org.springframework.stereotype.Service;

/**
 * PasswordResetService handles the "forgot password" flow: issuing a
 * short-lived reset token by email, then applying the new password once the
 * user completes the reset.
 */
@Service
public class PasswordResetService {

    private final UserRepository userRepository;
    private final TokenService tokenService;
    private final NotificationService notificationService;

    public PasswordResetService(UserRepository userRepository, TokenService tokenService, NotificationService notificationService) {
        this.userRepository = userRepository;
        this.tokenService = tokenService;
        this.notificationService = notificationService;
    }

    /**
     * Generates a reset token and emails it to the user so they can pick a
     * new password after forgetting the old one.
     */
    public String initiatePasswordReset(String email) {
        String token = tokenService.generateToken(email);
        notificationService.sendPasswordResetEmail(email);
        return token;
    }

    public String completePasswordReset(String token, String newPassword) {
        userRepository.findByEmail("placeholder");
        return "password updated";
    }
}
""",
    "src/main/java/com/example/auth/client/OidcClient.java": """\
package com.example.auth.client;

import org.springframework.stereotype.Component;

/**
 * OidcClient talks to the upstream OpenID Connect (OIDC) identity provider:
 * exchanging authorization codes for tokens during login, and propagating
 * end-session notifications during logout so the identity provider's own
 * session is terminated too, not just kenzo's local session.
 */
@Component
public class OidcClient {

    public String exchangeAuthorizationCode(String code) {
        return "access-token";
    }

    /**
     * Sends an OIDC end-session request to the identity provider. Must be
     * called whenever a user logs out locally, otherwise the identity
     * provider keeps believing the user's session is still active.
     */
    public void notifyIdentityProviderOfLogout(String sessionId) {
        // POST to the IdP's end_session_endpoint
    }
}
""",
    "src/main/java/com/example/auth/repository/SessionRepository.java": """\
package com.example.auth.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

/**
 * SessionRepository persists active user sessions.
 */
@Repository
public interface SessionRepository extends JpaRepository<Object, Long> {

    Object createSession(String userId);

    Object findActiveSession(String sessionId);

    /**
     * Marks a session as no longer valid, e.g. after the owning user signs
     * out or their session expires.
     */
    void invalidateSession(String sessionId);

    long countActiveSessions();
}
""",
    "src/main/java/com/example/user/controller/UserController.java": """\
package com.example.user.controller;

import com.example.user.service.UserService;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * UserController exposes CRUD endpoints for managing user accounts.
 */
@RestController
@RequestMapping("/api/users")
public class UserController {

    private final UserService userService;

    public UserController(UserService userService) {
        this.userService = userService;
    }

    @GetMapping("/{id}")
    public String getUser(String id) {
        return userService.getUserById(id);
    }

    @PostMapping
    public String createUser(String email) {
        return userService.createUser(email);
    }

    @PutMapping("/{id}")
    public String updateUser(String id) {
        return userService.updateUserProfile(id);
    }

    @DeleteMapping("/{id}")
    public String deleteUser(String id) {
        return userService.deactivateUser(id);
    }

    @GetMapping
    public String listUsers() {
        return "list of users";
    }
}
""",
    "src/main/java/com/example/user/service/UserService.java": """\
package com.example.user.service;

import com.example.user.repository.UserRepository;
import org.springframework.stereotype.Service;

/**
 * UserService contains business logic for creating, updating, and
 * deactivating user accounts, and checking whether a user has the
 * necessary permissions to perform an action.
 */
@Service
public class UserService {

    private final UserRepository userRepository;
    private final PermissionService permissionService;

    public UserService(UserRepository userRepository, PermissionService permissionService) {
        this.userRepository = userRepository;
        this.permissionService = permissionService;
    }

    public String getUserById(String id) {
        userRepository.findById(id);
        return "user " + id;
    }

    public String createUser(String email) {
        userRepository.existsByEmail(email);
        return "created";
    }

    public String updateUserProfile(String id) {
        return "updated";
    }

    /**
     * Deactivates a user account so they can no longer sign in, without
     * deleting their historical data.
     */
    public String deactivateUser(String id) {
        permissionService.revokeRole(id, "USER");
        return "deactivated";
    }
}
""",
    "src/main/java/com/example/user/service/PermissionService.java": """\
package com.example.user.service;

import com.example.user.repository.RoleRepository;
import org.springframework.stereotype.Service;

/**
 * PermissionService checks and manages what a user is allowed to do,
 * backed by a simple role-based access model.
 */
@Service
public class PermissionService {

    private final RoleRepository roleRepository;

    public PermissionService(RoleRepository roleRepository) {
        this.roleRepository = roleRepository;
    }

    public boolean checkPermission(String userId, String permission) {
        roleRepository.findAllByUserId(userId);
        return true;
    }

    public void grantRole(String userId, String role) {
        // add role
    }

    public void revokeRole(String userId, String role) {
        // remove role
    }
}
""",
    "src/main/java/com/example/user/repository/UserRepository.java": """\
package com.example.user.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface UserRepository extends JpaRepository<Object, Long> {

    Object findById(String id);

    Object findByEmail(String email);

    boolean existsByEmail(String email);
}
""",
    "src/main/java/com/example/user/repository/RoleRepository.java": """\
package com.example.user.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface RoleRepository extends JpaRepository<Object, Long> {

    Object findByName(String name);

    java.util.List<Object> findAllByUserId(String userId);
}
""",
    "src/main/java/com/example/billing/controller/InvoiceController.java": """\
package com.example.billing.controller;

import com.example.billing.service.BillingService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * InvoiceController exposes endpoints for generating and retrieving
 * customer invoices.
 */
@RestController
@RequestMapping("/api/billing/invoices")
public class InvoiceController {

    private final BillingService billingService;

    public InvoiceController(BillingService billingService) {
        this.billingService = billingService;
    }

    @PostMapping
    public String createInvoice(String orderId) {
        return billingService.generateInvoice(orderId);
    }

    @GetMapping("/{id}")
    public String getInvoice(String id) {
        return "invoice " + id;
    }

    @GetMapping
    public String listInvoices() {
        return "list of invoices";
    }
}
""",
    "src/main/java/com/example/billing/service/BillingService.java": """\
package com.example.billing.service;

import com.example.billing.repository.InvoiceRepository;
import org.springframework.stereotype.Service;

/**
 * BillingService generates invoices for completed orders and coordinates
 * charging the customer's payment method for the invoice total.
 */
@Service
public class BillingService {

    private final InvoiceRepository invoiceRepository;
    private final PaymentService paymentService;

    public BillingService(InvoiceRepository invoiceRepository, PaymentService paymentService) {
        this.invoiceRepository = invoiceRepository;
        this.paymentService = paymentService;
    }

    public String generateInvoice(String orderId) {
        invoiceRepository.findByOrderId(orderId);
        double total = calculateTotal(orderId);
        paymentService.chargeCustomer(orderId, total);
        return "invoice generated";
    }

    public double calculateTotal(String orderId) {
        return 100.0;
    }

    /**
     * Cancels an invoice and, if it was already paid, triggers a refund so
     * the customer isn't charged twice for the same order.
     */
    public String voidInvoice(String invoiceId) {
        paymentService.refundPayment(invoiceId);
        return "voided";
    }
}
""",
    "src/main/java/com/example/billing/service/PaymentService.java": """\
package com.example.billing.service;

import com.example.billing.client.PaymentGatewayClient;
import org.springframework.stereotype.Service;

/**
 * PaymentService talks to the payment gateway to charge customers and
 * process refunds, and retries payments that failed transiently.
 */
@Service
public class PaymentService {

    private final PaymentGatewayClient paymentGatewayClient;

    public PaymentService(PaymentGatewayClient paymentGatewayClient) {
        this.paymentGatewayClient = paymentGatewayClient;
    }

    public String chargeCustomer(String customerId, double amount) {
        paymentGatewayClient.submitCharge(customerId, amount);
        return "charged";
    }

    public String refundPayment(String paymentId) {
        paymentGatewayClient.submitRefund(paymentId);
        return "refunded";
    }

    /**
     * Retries a payment that previously failed, e.g. due to a transient
     * gateway timeout, without double-charging the customer.
     */
    public String retryFailedPayment(String paymentId) {
        return "retried";
    }
}
""",
    "src/main/java/com/example/billing/client/PaymentGatewayClient.java": """\
package com.example.billing.client;

import org.springframework.stereotype.Component;

/**
 * PaymentGatewayClient is the low-level client for the external card
 * payment gateway.
 */
@Component
public class PaymentGatewayClient {

    public String submitCharge(String customerId, double amount) {
        return "charge-ref";
    }

    public String submitRefund(String paymentId) {
        return "refund-ref";
    }
}
""",
    "src/main/java/com/example/billing/repository/InvoiceRepository.java": """\
package com.example.billing.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface InvoiceRepository extends JpaRepository<Object, Long> {

    Object findByOrderId(String orderId);

    java.util.List<Object> findByCustomerId(String customerId);
}
""",
    "src/main/java/com/example/notification/service/NotificationService.java": """\
package com.example.notification.service;

import com.example.notification.client.EmailClient;
import org.springframework.stereotype.Service;

/**
 * NotificationService sends transactional emails to users: welcome emails,
 * password reset emails, and invoice receipts.
 */
@Service
public class NotificationService {

    private final EmailClient emailClient;

    public NotificationService(EmailClient emailClient) {
        this.emailClient = emailClient;
    }

    public void sendWelcomeEmail(String userId) {
        emailClient.sendEmail(userId, "Welcome", "Welcome to the app");
    }

    /**
     * Emails the user a link to reset their forgotten password.
     */
    public void sendPasswordResetEmail(String email) {
        emailClient.sendEmail(email, "Reset your password", "Click here to reset");
    }

    public void sendInvoiceReceipt(String invoiceId) {
        emailClient.sendEmail("customer@example.com", "Your receipt", "Thanks for your payment");
    }
}
""",
    "src/main/java/com/example/notification/client/EmailClient.java": """\
package com.example.notification.client;

import org.springframework.stereotype.Component;

/**
 * EmailClient is the low-level client for the outbound email provider.
 */
@Component
public class EmailClient {

    public void sendEmail(String to, String subject, String body) {
        // send via SMTP provider
    }
}
""",
    "src/main/java/com/example/audit/service/AuditLogService.java": """\
package com.example.audit.service;

import com.example.audit.repository.AuditLogRepository;
import org.springframework.stereotype.Service;

/**
 * AuditLogService records security-relevant events (logins, logouts,
 * permission changes) for later review.
 */
@Service
public class AuditLogService {

    private final AuditLogRepository auditLogRepository;

    public AuditLogService(AuditLogRepository auditLogRepository) {
        this.auditLogRepository = auditLogRepository;
    }

    public void recordLoginEvent(String userId) {
        auditLogRepository.save(userId, "LOGIN");
    }

    /**
     * Records that a user signed out, for later security review.
     */
    public void recordLogoutEvent(String userId) {
        auditLogRepository.save(userId, "LOGOUT");
    }

    public void recordPermissionChange(String userId) {
        auditLogRepository.save(userId, "PERMISSION_CHANGE");
    }
}
""",
    "src/main/java/com/example/audit/repository/AuditLogRepository.java": """\
package com.example.audit.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface AuditLogRepository extends JpaRepository<Object, Long> {

    Object save(String userId, String eventType);

    java.util.List<Object> findByUserId(String userId);
}
""",
}


def write_fixture(root: Path) -> int:
    """Write every fixture file under `root`. Returns the file count."""
    for rel_path, content in _FIXTURE_FILES.items():
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return len(_FIXTURE_FILES)


# ---------------------------------------------------------------------------
# "Before Phase 5" baseline: exact -> substring -> signature-text keyword
# overlap. Operates on the SAME already-built SymbolIndex as the hybrid
# engine (fair comparison: same parsed data, different retrieval algorithm).
# ---------------------------------------------------------------------------


@dataclass
class _Entry:
    symbol: str  # "ClassName" or "ClassName.methodName" or "functionName"
    file: str
    chunk_type: str  # "class" | "method" | "function"
    name: str  # simple name, used for substring/keyword scoring
    qualified_name: str
    signature: str  # rendered structural signature text (NO comments/body)


@dataclass
class BaselineHit:
    symbol: str
    file: str
    chunk_type: str
    match_kind: str  # "exact" | "substring" | "keyword"
    score: float


def _render_class_signature(cls: ClassSymbol) -> str:
    return "\n".join(
        [
            f"{cls.kind} {cls.name}",
            f"stereotype: {cls.stereotype or 'none'}",
            f"extends: {cls.extends or 'none'}",
            f"implements: {', '.join(cls.implements) or 'none'}",
            f"annotations: {', '.join(cls.annotations) or 'none'}",
            f"dependencies: {', '.join(cls.dependencies) or 'none'}",
            f"endpoints: {', '.join(cls.endpoints) or 'none'}",
            f"methods: {', '.join(m.name for m in cls.methods) or 'none'}",
        ]
    )


def _render_function_signature(qualified_name: str, fn: FunctionSymbol) -> str:
    params = ", ".join((f"{p.name}: {p.type}" if p.type else p.name) for p in fn.parameters)
    kind = "method" if fn.is_method else "function"
    return "\n".join(
        [
            f"{kind} {qualified_name}",
            f"annotations: {', '.join(fn.annotations) or 'none'}",
            f"endpoint: {fn.endpoint or 'none'}",
            f"parameters: {params or 'none'}",
            f"return type: {fn.return_type or 'none'}",
        ]
    )


def collect_baseline_entries(symbol_index: SymbolIndex) -> list[_Entry]:
    entries: list[_Entry] = []
    for file_path, cls in symbol_index.all_classes():
        entries.append(
            _Entry(
                symbol=cls.name,
                file=file_path,
                chunk_type="class",
                name=cls.name,
                qualified_name=cls.qualified_name or cls.name,
                signature=_render_class_signature(cls),
            )
        )
        for method in cls.methods:
            qualified = method.qualified_name or f"{cls.name}.{method.name}"
            entries.append(
                _Entry(
                    symbol=f"{cls.name}.{method.name}",
                    file=file_path,
                    chunk_type="method",
                    name=method.name,
                    qualified_name=qualified,
                    signature=_render_function_signature(qualified, method),
                )
            )
    for file_path, fn in symbol_index.all_functions():
        qualified = fn.qualified_name or fn.name
        entries.append(
            _Entry(
                symbol=fn.name,
                file=file_path,
                chunk_type="function",
                name=fn.name,
                qualified_name=qualified,
                signature=_render_function_signature(qualified, fn),
            )
        )
    return entries


def baseline_search(entries: list[_Entry], query: str, top_k: int = 3) -> list[BaselineHit]:
    """(a) exact -> (b) substring -> (c) signature keyword-overlap. Each step
    is used ONLY if the previous step found nothing -- this is a strict
    waterfall, exactly mirroring how the real `find_symbol` tool prefers an
    exact match over its substring fallback, plus one extra fallback layer.
    """
    normalized_query = query.strip().lower()
    if not normalized_query:
        return []

    # (a) exact
    exact = [e for e in entries if e.name.lower() == normalized_query or e.qualified_name.lower() == normalized_query]
    if exact:
        ranked = sorted(exact, key=lambda e: (e.file, e.symbol))
        return [BaselineHit(e.symbol, e.file, e.chunk_type, "exact", 1.0) for e in ranked[:top_k]]

    # (b) substring
    substring = [
        e for e in entries if normalized_query in e.name.lower() or normalized_query in e.qualified_name.lower()
    ]
    if substring:

        def _sub_score(e: _Entry) -> float:
            name_len = len(e.name) or 1
            return min(1.0, len(normalized_query) / name_len)

        ranked = sorted(substring, key=lambda e: (-_sub_score(e), e.file, e.symbol))
        return [BaselineHit(e.symbol, e.file, e.chunk_type, "substring", round(_sub_score(e), 3)) for e in ranked[:top_k]]

    # (c) signature keyword/token overlap (same tokenizer as the hybrid
    # engine's own "keyword" signal, for an apples-to-apples comparison of
    # what pure token-overlap can do with less text to work with).
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []
    scored: list[tuple[float, _Entry]] = []
    for e in entries:
        sig_tokens = set(_tokenize(e.signature))
        matched = sum(1 for t in query_tokens if t in sig_tokens)
        if matched > 0:
            scored.append((matched / len(query_tokens), e))
    if not scored:
        return []
    scored.sort(key=lambda pair: (-pair[0], pair[1].file, pair[1].symbol))
    return [BaselineHit(e.symbol, e.file, e.chunk_type, "keyword", round(score, 3)) for score, e in scored[:top_k]]


# ---------------------------------------------------------------------------
# Query set
# ---------------------------------------------------------------------------


@dataclass
class Query:
    category: str
    text: str
    # Lowercase substrings of a result's `symbol` that count as "relevant"
    # for this query -- MY OWN judgment of ground truth (like a benchmark's
    # qrels), used only to compute the printed verdict/tally, never fed into
    # either search algorithm.
    relevant_substrings: list[str]


QUERIES: list[Query] = [
    # 1. Exact name queries
    Query("1-exact", "AuthService", ["authservice"]),
    Query("1-exact", "invalidateSession", ["invalidatesession"]),
    # 2. Partial/substring queries
    Query("2-partial", "Permission", ["permission"]),
    Query("2-partial", "Invoice", ["invoice"]),
    # 3. Conceptual queries (different words, some lexical/character overlap)
    Query(
        "3-conceptual",
        "sign the user out",
        ["authservice.logout", "authcontroller.logout", "sessionrepository.invalidatesession", "tokenservice.revoketoken"],
    ),
    Query("3-conceptual", "reset a forgotten password", ["passwordresetservice", "passwordresetcontroller"]),
    Query("3-conceptual", "charge a customer's card", ["paymentservice.chargecustomer", "paymentgatewayclient.submitcharge"]),
    # 4. Problem/symptom queries
    Query(
        "4-symptom",
        "users can't sign out",
        [
            "authservice.logout",
            "authcontroller.logout",
            "sessionrepository.invalidatesession",
            "tokenservice.revoketoken",
            "oidcclient",
        ],
    ),
    Query("4-symptom", "OIDC logout not called", ["oidcclient"]),
    Query(
        "4-symptom",
        "password reset emails never arrive",
        ["notificationservice.sendpasswordresetemail", "emailclient.sendemail", "passwordresetservice"],
    ),
    Query(
        "4-symptom",
        "customers charged twice for the same invoice",
        [
            "billingservice.voidinvoice",
            "paymentservice.refundpayment",
            "paymentservice.retryfailedpayment",
            "paymentgatewayclient.submitrefund",
        ],
    ),
]


def _is_relevant(symbol: str, relevant_substrings: list[str]) -> bool:
    s = symbol.lower()
    return any(sub in s for sub in relevant_substrings)


def _top3_relevant(symbols: list[str], relevant_substrings: list[str]) -> bool:
    return any(_is_relevant(s, relevant_substrings) for s in symbols[:3])


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------


def _fmt_baseline(hits: list[BaselineHit]) -> list[str]:
    if not hits:
        return ["  (no results)"]
    return [f"  [{h.match_kind:<9}] {h.symbol}  ({h.chunk_type}, {h.file})  score={h.score}" for h in hits]


def _fmt_hybrid(results: list[SearchResult]) -> list[str]:
    if not results:
        return ["  (no results)"]
    lines = []
    for r in results:
        sig = ", ".join(f"{k}={v:.3f}" for k, v in r.signals.items())
        lines.append(f"  [score={r.score:.3f}] {r.symbol}  ({r.chunk_type}, {r.file})  signals: {{{sig}}}")
    return lines


def main() -> None:
    tmp_root = Path(tempfile.mkdtemp(prefix="kenzo_retrieval_bench_"))
    print(f"Fixture directory: {tmp_root}")

    try:
        n_files = write_fixture(tmp_root)
        print(f"Wrote {n_files} Java files (21 classes across auth/user/billing/notification/audit concerns).\n")

        # --- Phase 4 structural intelligence (shared by both approaches) ---
        symbol_index = SymbolIndex(str(tmp_root))
        symbol_index.build(max_files=1000)
        stats = symbol_index.stats()
        n_methods = sum(len(cls.methods) for _, cls in symbol_index.all_classes())
        print(
            f"SymbolIndex.build(): {stats['file_count']} files, {stats['class_count']} classes, "
            f"{n_methods} methods, {stats['function_count']} top-level functions "
            f"(0 is expected/correct here -- Java methods always live inside a class, so "
            f"SymbolIndex.stats()'s 'function_count' -- top-level functions only -- is 0 for an "
            f"all-Java fixture like this one; the real per-class method count is {n_methods})"
        )

        reference_graph = ReferenceGraph(symbol_index)
        reference_graph.build()
        dependency_graph = DependencyGraph(symbol_index)
        dependency_graph.build()
        n_callee_edges = sum(len(v) for v in reference_graph.callees.values())
        print(
            f"ReferenceGraph.build(): {n_callee_edges} resolved call edges. "
            f"DependencyGraph.build(): {len(dependency_graph.class_deps)} classes with dependency edges."
        )

        # --- Phase 5 semantic memory (real production wiring, see
        # kenzo/core/indexing_service.py) ---
        embedding_router = EmbeddingRouter(EmbeddingConfig())  # default: provider="local" -> HashingEmbeddingProvider
        vector_store = QdrantStore(
            collection_name="benchmark_retrieval", dimensions=embedding_router.dimensions, url=":memory:"
        )
        semantic_indexer = SemanticIndexer(str(tmp_root), symbol_index, embedding_router, vector_store)

        t0 = time.perf_counter()
        n_chunks = semantic_indexer.index_all()
        index_seconds = time.perf_counter() - t0
        print(
            f"\nSemanticIndexer.index_all(): {index_seconds:.3f}s for {n_chunks} chunks "
            f"(local HashingEmbeddingProvider, embedded Qdrant ':memory:' store) "
            f"-- for context, scripts/benchmark_indexing.py separately measures Phase 4's raw "
            f"indexing SPEED at much larger synthetic-repo scales; this single data point is just "
            f"'is Phase 5's extra embed+store pass fast enough to be interactive' at this fixture's size."
        )

        engine = CodeSearchEngine(symbol_index, reference_graph, dependency_graph, embedding_router, vector_store)
        baseline_entries = collect_baseline_entries(symbol_index)

        print("\n" + "=" * 100)
        print("PER-QUERY RESULTS")
        print("=" * 100)

        tally = {"hybrid_only": 0, "both": 0, "neither": 0, "baseline_only": 0}
        per_category_hybrid_only: dict[str, int] = {}

        for i, q in enumerate(QUERIES, start=1):
            baseline_hits = baseline_search(baseline_entries, q.text, top_k=3)
            hybrid_results = engine.search(q.text, top_k=3)

            baseline_relevant = _top3_relevant([h.symbol for h in baseline_hits], q.relevant_substrings)
            hybrid_relevant = _top3_relevant([r.symbol for r in hybrid_results], q.relevant_substrings)

            if hybrid_relevant and not baseline_relevant:
                verdict = "HYBRID WINS -- found a relevant result baseline missed."
                tally["hybrid_only"] += 1
                per_category_hybrid_only[q.category] = per_category_hybrid_only.get(q.category, 0) + 1
            elif hybrid_relevant and baseline_relevant:
                verdict = "TIE -- both surfaced a relevant result in their top-3."
                tally["both"] += 1
            elif baseline_relevant and not hybrid_relevant:
                verdict = "BASELINE WINS (unexpected) -- baseline found a relevant result, hybrid did not."
                tally["baseline_only"] += 1
            else:
                verdict = "NEITHER found a relevant result in their top-3."
                tally["neither"] += 1

            print(f"\n[{i}] ({q.category}) Query: \"{q.text}\"")
            print(f"  ground-truth relevant substrings: {q.relevant_substrings}")
            print(" Baseline (Before Phase 5) top-3:")
            for line in _fmt_baseline(baseline_hits):
                print(line)
            print(" Hybrid CodeSearchEngine (After Phase 5) top-3:")
            for line in _fmt_hybrid(hybrid_results):
                print(line)
            print(f" VERDICT: {verdict}")

        total = len(QUERIES)
        print("\n" + "=" * 100)
        print("SUMMARY")
        print("=" * 100)
        print(f"Total queries: {total}")
        print(f"  Hybrid found a relevant result baseline missed: {tally['hybrid_only']}/{total}")
        print(f"  Both found a relevant result:                   {tally['both']}/{total}")
        print(f"  Baseline found a relevant result hybrid missed: {tally['baseline_only']}/{total}")
        print(f"  Neither found a relevant result:                {tally['neither']}/{total}")
        print(f"\nBy category, queries where hybrid won outright: {per_category_hybrid_only}")
        print(f"\nIndexing-time data point: SemanticIndexer.index_all() took {index_seconds:.3f}s for "
              f"{n_chunks} chunks across {n_files} files on this machine.")

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
        print(f"\nCleaned up fixture directory: {tmp_root}")


if __name__ == "__main__":
    main()
