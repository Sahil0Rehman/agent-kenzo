"""Phase 8 "Vision Debugging Engine": screenshots in, structured
understanding out, fed into the SAME `AgentController` loop every other
phase already drives -- vision is a new INPUT modality, not a new agent.

`kenzo.vision.models.VisionAnalysis` is the shared contract:
`ModelRouter.analyze_image()` (kenzo/models/router.py) produces it,
`AgentController.set_vision_context()` (kenzo/core/controller.py) consumes
it as priority context, exactly mirroring Phase 7's `IdeContext` pattern.
"""
