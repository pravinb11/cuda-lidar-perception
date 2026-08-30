#include <stdio.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "pointcloud_loader.h"

bool loadPCD(
    const char *filename,
    std::vector<Point>& points)
{
    std::ifstream file(filename);

    if (!file.is_open())
    {
        printf(
            "Failed to open PCD file: %s\n",
            filename);

        return false;
    }

    std::string line;

    bool dataStarted = false;

    while (std::getline(file, line))
    {
        if (line.empty())
        {
            continue;
        }

        // ----------------------------
        // Header
        // ----------------------------

        if (!dataStarted)
        {
            if (line.rfind("DATA", 0) == 0)
            {
                std::stringstream ss(line);

                std::string keyword;
                std::string format;

                ss >> keyword >> format;

                if (format != "ascii")
                {
                    printf(
                        "Only ASCII PCD is supported\n");

                    return false;
                }

                dataStarted = true;
            }

            continue;
        }

        // ----------------------------
        // Point data
        // ----------------------------

        std::stringstream ss(line);

        Point p;

        if (!(ss >> p.x >> p.y >> p.z))
        {
            printf(
                "Invalid point data: %s\n",
                line.c_str());

            return false;
        }

        points.push_back(p);
    }

    printf(
        "Loaded %zu points from %s\n",
        points.size(),
        filename);

    return true;
}