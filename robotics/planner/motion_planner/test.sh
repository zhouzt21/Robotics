version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2 | tr -d '.') 
cd ~/RealRobot/third_party/MPlib
bash dev/build_wheels_v2.sh --py $version
pip uninstall mplib -y
pip install dist/mplib-0.0.9-cp$version-cp$version-linux_x86_64.whl
cd ~/RealRobot/realbot/agent/motion_planner
python3 tester.py
