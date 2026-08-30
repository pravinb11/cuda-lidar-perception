#pragma once

#include <vector>

#include "point.h"

bool loadPCD(
    const char *filename, std::vector <Point>& points);