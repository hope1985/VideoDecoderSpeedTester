#include "YUVMemoryStream.h"

    void YUVMemoryStream::addFrame(const uint8_t* frameData, size_t frameSize) {
        buffer.insert(buffer.end(), frameData, frameData + frameSize);
    }

    void YUVMemoryStream::writeToFile() {

        if (buffer.size() > 0)
        {
            outFile->write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
            buffer.clear();
        }
    }

    void YUVMemoryStream::openFile(const std::string& filename) {

       
		buffer = std::vector<uint8_t>();
        outFile = new std::ofstream(filename, std::ios::binary);
        if (!outFile) {
            throw std::runtime_error("Failed to open file for writing: " + filename);
        }

    }

    size_t YUVMemoryStream::totalSize()  {
        return buffer.size();
    }




