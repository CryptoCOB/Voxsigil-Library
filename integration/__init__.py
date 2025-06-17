# Integration Components
# Contains integration layers and connectors

try:
    from .real_supervisor_connector import RealSupervisorConnector
except ImportError:
    # Create a stub if import fails
    class RealSupervisorConnector:
        def __init__(self, *args, **kwargs):
            pass

        def connect(self):
            return True

        def disconnect(self):
            pass


try:
    from .vanta_registration import VantaRegistration
except ImportError:
    VantaRegistration = None

try:
    from .voxsigil_integration import VoxSigilIntegrationManager as VoxSigilIntegration
except ImportError:
    VoxSigilIntegration = None

__all__ = ["RealSupervisorConnector", "VantaRegistration", "VoxSigilIntegration"]
