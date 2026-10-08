from zhinst.toolkit import Session
from zhinst.core.errors import DeviceInUseError

DEVICE_ID = "DEV4388"

session = Session("localhost")

print("Connected to Data Server:")
print(session)

print("\nVisible devices:")
try:
    print(session.devices.visible())
except Exception as exc:
    print(f"Could not read visible devices: {exc}")

print("\nConnected devices:")
try:
    print(session.devices.connected())
except Exception as exc:
    print(f"Could not read connected devices: {exc}")

print("\nTrying to connect to device...")
try:
    device = session.connect_device(DEVICE_ID, interface="1GbE")
    print(f"Connected to {DEVICE_ID}")
    print(device)
except DeviceInUseError as exc:
    print("\nDEVICE IS ALREADY IN USE")
    print(exc)
    print(
        "\nClose/disconnect LabOne GUI and any other Python/LabVIEW sessions, "
        "then restart the Zurich Data Server if needed."
    )
except Exception as exc:
    print("\nConnection failed with different error:")
    print(type(exc).__name__)
    print(exc)