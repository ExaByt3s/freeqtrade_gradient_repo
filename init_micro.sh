INSTALL_PREFIX='/usr'
INSTALL_PATH="${INSTALL_PREFIX}/bin"
su - root -c "mkdir --parents ${INSTALL_PATH} && cd ${INSTALL_PATH} && wget -O - https://getmic.ro | sh"
