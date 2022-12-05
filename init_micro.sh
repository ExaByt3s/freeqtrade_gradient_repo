su - root -c "cd /usr/bin; wget -O- https://getmic.ro | GETMICRO_REGISTER=y sh"
cp /usr/bin/micro freqtrade/freqtrade/.env/bin/micro
