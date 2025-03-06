#include <Arduino.h>
#include <mbed_stats.h>

#define T2D_SOM_TEST_DOUBLE

#ifdef BASIC_TEST_FLOAT
#include "examples/float/BasicExample.h"
#endif
#ifdef BASIC_WITH_ERROR_TEST_FLOAT
#include "examples/float/BasicTestWithError.h"
#endif
#ifdef T2D_SOM_TEST_FLOAT
#include "examples/float/2dSomExample.h"
#endif
#ifdef T3D_HCM_TEST_FLOAT
#include "examples/float/3DTest.h"
#endif

#ifdef T2D_SOM_TEST_DOUBLE
#include "examples/double/2dSomExample.h"
#endif


const String title = "   _____ _____ _____ _____ ______            _   \n"
                     "  / ____|  __ \\_   _/ ____|  ____|          | |  \n"
                     " | (___ | |__) || || |    | |__   _ __   ___| |_ \n"
                     "  \\___ \\|  ___/ | || |    |  __| | '_ \\ / _ \\ __|\n"
                     "  ____) | |    _| || |____| |____| | | |  __/ |_ \n"
                     " |_____/|_|   |_____\\_____|______|_| |_|\\___|\\__|\n"
                     "                                                 \n";
extern "C" char *sbrk(int incr);

int freeRam() {
    char top;
    return &top - reinterpret_cast<char *>(sbrk(0));
}


void display_freeram() {
    Serial.print(F("- SRAM left: "));
    Serial.println(freeRam());

    mbed_stats_heap_t heap_stats;
    mbed_stats_heap_get(&heap_stats);
    Serial.println(heap_stats.current_size);
    Serial.println(heap_stats.reserved_size);
}


void setup() {
    Serial.begin(9600);
    delay(5000);
    String st = "test";
    Serial.print(title);

#ifdef BASIC_TEST_FLOAT
    runTest();
#endif
#ifdef T2D_SOM_TEST_FLOAT
    runTest();
#endif
#ifdef BASIC_WITH_ERROR_TEST_FLOAT
    runTest();
#endif
#ifdef T3D_HCM_TEST_FLOAT
    runTest();
#endif

#ifdef T2D_SOM_TEST_DOUBLE
    runTest();
#endif


    randomSeed(42);
    pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
    digitalWrite(LED_BUILTIN, HIGH);  // turn the LED on (HIGH is the voltage level)
    delay(1000);                      // wait for a second
    digitalWrite(LED_BUILTIN, LOW);   // turn the LED off by making the voltage LOW
    delay(1000);                      // wait for a second
}