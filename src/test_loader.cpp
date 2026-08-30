#include <stdio.h>

#include <vector>

#include "pointcloud_loader.h"

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf(
            "Usage: test_loader <file.pcd>\n");

        return 1;
    }

    std::vector<Point> points;

    if (!loadPCD(
            argv[1],
            points))
    {
        return 1;
    }

    printf(
        "Number of points: %zu\n",
        points.size());

    int printCount =
        points.size() < 5
        ? points.size()
        : 5;

    for (int i = 0;
         i < printCount;
         i++)
    {
        printf(
            "P%d = (%f, %f, %f)\n",
            i,
            points[i].x,
            points[i].y,
            points[i].z);
    }

    return 0;
}