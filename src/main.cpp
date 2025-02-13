#include <Arduino.h>
#include <mbed_stats.h>

#define BASIC_TEST

#ifdef BASIC_TEST
#include "examples/BasicExample.h"
#endif
#ifdef T2D_SOM_TEST
#include "examples/2dSomExample.h"
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

#ifdef BASIC_TEST
    runTest();
#endif
#ifdef T2D_SOM_TEST
    runTest();
#endif

    randomSeed(42);
}

void loop() {
    /*
    Serial.println("Doing stuff");
    float rnd = (float) random(-100000, 100000) / (float) 100000.0;
    Serial.println(rnd, 20);
    std::map<uint8_t, std::vector<float>> inputValues{{0, {rnd}}};
    auto start = millis();
    auto predicted = spicenet.tryDecode(1, inputValues);
    auto end = millis();
    auto real = pow(rnd, 3);
    Serial.print("Decoding millis: ");
    Serial.println(end - start);
    Serial.print(predicted, 20);
    Serial.print(" real: ");
    Serial.println(real, 20);
    Serial.print("bias: ");
    Serial.println(predicted - real, 20);
    display_freeram();
    delay(1000);
     */
}