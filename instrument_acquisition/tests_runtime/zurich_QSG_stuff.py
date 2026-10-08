from zhinst.toolkit import Session
session = Session("localhost")
#print(session)
# prints DataServerSession(localhost:8004)
print(list(session.child_nodes(recursive=True, leavesonly=True)))
#prints [/zi/config/open, /zi/config/port, /zi/about/revision, /zi/about/version, /zi/about/fullversion, /zi/about/commit, /zi/about/copyright, /zi/about/dataserver, /zi/about/fwrevision, /zi/debug/logpath, /zi/debug/level, /zi/debug/log, /zi/clockbase, /zi/devices/visible, /zi/devices/connected, /zi/mds/groups/0/devices, /zi/mds/groups/0/status, /zi/mds/groups/0/locked, /zi/mds/groups/0/keepalive]
# we get errors here:
device = session.connect_device("DEV4388")
""" Traceback (most recent call last):
  File "C:\Users\rackmcd\Desktop\mcd_python_aquisition\tests_runtime\zurich_QSG_stuff.py", line 7, in <module>
    device = session.connect_device("DEV4388")
  File "C:\Users\rackmcd\Desktop\mcd_python_aquisition\.venv\Lib\site-packages\zhinst\toolkit\session.py", line 939, in connect_device
    self._daq_server.connectDevice(serial, interface)  # type: ignore[arg-type]
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^
zhinst.core._core.errors.DeviceInUseError: Device 'DEV4388' is already in use with status: In use. [zi:api:32789] """ 
timestamp = device.status.time() 
print(timestamp)