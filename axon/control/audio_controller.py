import ctypes
from typing import Optional
from ..utils.logger import logger

# Import pycaw inside class to prevent crash on non-Windows platforms
try:
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL, CoInitialize, CoUninitialize
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    PYCAW_AVAILABLE = True
except ImportError:
    PYCAW_AVAILABLE = False


class AudioController:
    """Controls OS system volume and mute states on Windows using pycaw."""
    def __init__(self):
        self.volume_interface = None
        if PYCAW_AVAILABLE:
            self._init_audio()
        else:
            logger.warning("pycaw or comtypes is not available. Audio control will be disabled.")

    def _init_audio(self):
        """Initializes the Windows core audio endpoint interfaces."""
        try:
            # Initialize COM libraries for this thread
            CoInitialize()
            devices = AudioUtilities.GetSpeakers()
            if devices is None:
                logger.error("No speakers detected.")
                return
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self.volume_interface = cast(interface, POINTER(IAudioEndpointVolume))
            logger.info("pycaw audio interface successfully initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize pycaw volume control: {e}")

    def get_volume(self) -> float:
        """Returns volume level as float between 0.0 (mute) and 1.0 (max)."""
        if not self.volume_interface:
            return 0.0
        try:
            CoInitialize()
            return self.volume_interface.GetMasterVolumeLevelScalar()
        except Exception as e:
            logger.error(f"Error getting volume: {e}")
            return 0.0

    def set_volume(self, level: float):
        """Sets master volume. level must be between 0.0 and 1.0."""
        if not self.volume_interface:
            return
        try:
            CoInitialize()
            level = max(0.0, min(level, 1.0))
            self.volume_interface.SetMasterVolumeLevelScalar(level, None)
            logger.info(f"System volume set to: {int(level * 100)}%")
        except Exception as e:
            logger.error(f"Error setting volume: {e}")

    def volume_up(self, step: float = 0.05):
        """Increases volume by step amount."""
        curr = self.get_volume()
        self.set_volume(curr + step)

    def volume_down(self, step: float = 0.05):
        """Decreases volume by step amount."""
        curr = self.get_volume()
        self.set_volume(curr - step)

    def toggle_mute(self) -> bool:
        """Toggles the system mute state. Returns the new mute status."""
        if not self.volume_interface:
            return False
        try:
            CoInitialize()
            is_muted = self.volume_interface.GetMute()
            new_mute = not is_muted
            self.volume_interface.SetMute(new_mute, None)
            logger.info(f"System volume mute toggled: {new_mute}")
            return new_mute
        except Exception as e:
            logger.error(f"Error toggling mute: {e}")
            return False

    def is_muted(self) -> bool:
        if not self.volume_interface:
            return False
        try:
            CoInitialize()
            return bool(self.volume_interface.GetMute())
        except Exception as e:
            logger.error(f"Error checking mute state: {e}")
            return False

    def cleanup(self):
        """Releases COM objects on thread exit."""
        try:
            CoUninitialize()
        except Exception:
            pass

    def __repr__(self) -> str:
        vol = int(self.get_volume() * 100) if self.volume_interface else 0
        muted = self.is_muted() if self.volume_interface else False
        return f"AudioController(vol={vol}%, muted={muted})"
