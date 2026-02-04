#pragma once
#include "config.h"

IntegerWord add_mod(IntegerWord a, IntegerWord b, IntegerWord m); //(a + start) mod m
IntegerWord mul_mod(IntegerWord a, IntegerWord b, IntegerWord m); //(a * start) mod m
#define times_word(x, mod) mul_mod(x, -mod, mod) //(a * w) mod m
