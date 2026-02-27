#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <stdexcept>
class YUVMemoryStream {
public:
    void addFrame(const uint8_t* frameData, size_t frameSize);

    void writeToFile();

    void openFile(const std::string& filename); 

    size_t totalSize();

private:
    std::vector<uint8_t> buffer;
    std::ofstream* outFile = nullptr;
};
