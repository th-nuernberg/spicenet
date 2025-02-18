//
// Created by Fabian on 12.02.2025.
//

#ifndef SPICENET_SPICENETLOGGING_H
#define SPICENET_SPICENETLOGGING_H

#if defined(ARDUINO)
//#define SPICENET_LOGGING
#endif

#ifdef SPICENET_LOGGING
#define LOG(M) \
    Serial.print('['); \
    Serial.print(__FUNCTION__); \
    Serial.print("]: "); \
    Serial.print(M)

#define LOG_LN(M) \
    Serial.print('['); \
    Serial.print(__FUNCTION__); \
    Serial.print("]: "); \
    Serial.println(M)
#endif

#endif //SPICENET_SPICENETLOGGING_H
