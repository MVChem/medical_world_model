// Streaming CSV boundary scanner. Emits only patient IDs, byte spans and counts.
// No clinical fields are written. Quoted newlines, escaped quotes, CRLF, UTF-8,
// disjoint patient runs and a final record without a newline are supported.
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

class Scanner {
    int owner_column, columns, column = 0;
    bool header = true, quoted = false, after_quote = false;
    bool field_start = true, active = false, skip_lf = false;
    std::string owner, previous;
    uint64_t row_start = 0, span_start = 0, span_end = 0, span_rows = 0;
public:
    uint64_t rows = 0, header_end = 0;
    Scanner(int owner_column, int columns) : owner_column(owner_column), columns(columns) {}
    void emit() {
        if (span_rows && owner_column >= 0)
            std::cout << previous << '\t' << span_start << '\t' << span_end << '\t' << span_rows << '\n';
    }
    void character(char c) {
        if (!header && column == owner_column) owner.push_back(c);
        field_start = false;
        active = true;
    }
    void finish(uint64_t end) {
        if (active) {
            if (column + 1 != columns)
                throw std::runtime_error("CSV field count mismatch at byte " + std::to_string(row_start));
            if (header) {
                header = false;
                header_end = end;
            } else {
                if (owner_column >= 0 && (owner.empty() || !std::all_of(owner.begin(), owner.end(), [](char c) { return c >= '0' && c <= '9'; })))
                    throw std::runtime_error("Missing/invalid subject_id at byte " + std::to_string(row_start));
                if (!span_rows || owner != previous) {
                    emit();
                    previous = owner;
                    span_start = row_start;
                    span_rows = 0;
                }
                ++span_rows;
                ++rows;
                span_end = end;
            }
        }
        owner.clear(); column = 0; field_start = true;
        after_quote = false; active = false; row_start = end;
    }
    void feed(const char* data, size_t length, uint64_t offset) {
        for (size_t i = 0; i < length; ++i) {
            const char c = data[i];
            const uint64_t position = offset + i;
            if (position == 0 && length >= 3 &&
                static_cast<unsigned char>(data[0]) == 0xef &&
                static_cast<unsigned char>(data[1]) == 0xbb &&
                static_cast<unsigned char>(data[2]) == 0xbf) { i = 2; continue; }
            if (skip_lf) {
                skip_lf = false;
                if (c == '\n') {
                    if (span_end == position) span_end = position + 1;
                    if (header_end == position) header_end = position + 1;
                    row_start = position + 1;
                    continue;
                }
            }
            if (quoted) {
                if (c == '"') { quoted = false; after_quote = true; }
                else character(c);
                continue;
            }
            if (after_quote && c == '"') {
                character('"'); quoted = true; after_quote = false; continue;
            }
            after_quote = false;
            if (c == ',' ) { ++column; field_start = true; active = true; }
            else if (c == '\n' || c == '\r') {
                finish(position + 1);
                skip_lf = c == '\r';
            } else if (field_start && c == '"') {
                quoted = true; field_start = false; active = true;
            } else character(c);
        }
    }
    void done(uint64_t size) {
        if (quoted) throw std::runtime_error("Unterminated quoted CSV field");
        if (active) finish(size);
        if (header) throw std::runtime_error("Missing CSV header");
        emit();
        std::cout << "#summary\t" << rows << '\t' << header_end << '\t' << size << '\n';
    }
};

int main(int argc, char** argv) {
    try {
        if (argc != 5) throw std::runtime_error("Usage: csv_spans FILE SUBJECT_COLUMN COLUMN_COUNT BLOCK_BYTES");
        const int column = std::stoi(argv[2]), columns = std::stoi(argv[3]);
        const size_t block = std::stoull(argv[4]);
        if (columns <= 0 || column < -1 || column >= columns || block < 4)
            throw std::runtime_error("Invalid scanner arguments");
        std::ifstream input(argv[1], std::ios::binary);
        if (!input) throw std::runtime_error("Cannot open source CSV");
        std::ios::sync_with_stdio(false);
        Scanner scanner(column, columns);
        std::vector<char> buffer(block);
        uint64_t offset = 0;
        while (input.read(buffer.data(), buffer.size()) || input.gcount()) {
            const auto n = static_cast<size_t>(input.gcount());
            scanner.feed(buffer.data(), n, offset);
            offset += n;
        }
        if (!input.eof()) throw std::runtime_error("Source CSV read failed");
        scanner.done(offset);
        if (!std::cout) throw std::runtime_error("Index output failed");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
