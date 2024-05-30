# Notes on virtual screen on remote machine 



install
```sudo apt-get install xserver-xorg-video-dummy```

write config file by
```sudo vim /etc/X11/xorg.conf```

```bash
Section "Device"
    Identifier  "Configured Video Device"
    Driver      "dummy"
    # Default is 4MiB, this sets it to 16MiB
    VideoRam    16384
EndSection

Section "Monitor"
    Identifier  "Configured Monitor"
    HorizSync 31.5-48.5
    VertRefresh 50-70
EndSection

Section "Screen"
    Identifier  "Default Screen"
    Monitor     "Configured Monitor"
    Device      "Configured Video Device"
    DefaultDepth 24
    SubSection "Display"
    Depth 24
    Modes "1024x800"
    EndSubSection
EndSection
```

Then 
```
(sudo) X :1 -config xorg.conf
```

install lightdm
```sudo apt-get install lightdm```

start lightdm
```sudo service lightdm start```

open gnome
```DISPLAY=:1 gnome-session```

setup vnctight
```sudo apt-get install x11vnc```

start vnc server
```x11vnc -display :1 -forever -localhost -rfbport 5900```