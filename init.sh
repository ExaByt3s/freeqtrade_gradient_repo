# . /opt/poplar/enable.sh
# /root/.gitconfig
sh init_apt.sh
echo /notebooks/.usr/lib > /etc/ld.so.conf.d/notebooks.conf
ldconfig
ln -sf /notebooks/setting/micro/bindings.json ~/.config/micro/bindings.json
ln -sf /notebooks/setting/micro/settings.json ~/.config/micro/settings.json
yash
