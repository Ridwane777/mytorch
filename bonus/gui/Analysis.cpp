#include "Analysis.hpp"

#include <cstdio>

#include "Oracle.hpp"
#include "Saliency.hpp"
#include "my_torch/Error.hpp"
#include "my_torch/Metrics.hpp"

namespace gui {

namespace {

int hexDigit(char c)
{
    if (c >= '0' && c <= '9')
        return c - '0';
    if (c >= 'a' && c <= 'f')
        return c - 'a' + 10;
    if (c >= 'A' && c <= 'F')
        return c - 'A' + 10;
    return -1;
}

} // namespace

std::string urlDecode(std::string_view text)
{
    std::string out;

    for (std::size_t i = 0; i < text.size(); i++) {
        if (text[i] == '+') {
            out += ' ';
        } else if (text[i] == '%') {
            if (i + 2 >= text.size() || hexDigit(text[i + 1]) < 0 || hexDigit(text[i + 2]) < 0)
                throw my_torch::Error("malformed escape in the query");
            out += static_cast<char>(hexDigit(text[i + 1]) * 16 + hexDigit(text[i + 2]));
            i += 2;
        } else {
            out += text[i];
        }
    }
    return out;
}

std::string queryValue(std::string_view query, std::string_view key)
{
    while (!query.empty()) {
        const std::size_t amp = query.find('&');
        const std::string_view pair = query.substr(0, amp);
        const std::size_t eq = pair.find('=');
        if (pair.substr(0, eq) == key)
            return eq == std::string_view::npos ? "" : urlDecode(pair.substr(eq + 1));
        query = amp == std::string_view::npos ? "" : query.substr(amp + 1);
    }
    return "";
}

std::string jsonEscape(std::string_view text)
{
    std::string out;

    for (const char c : text) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += c;
        } else if (static_cast<unsigned char>(c) < 0x20) {
            char escaped[8];
            std::snprintf(escaped, sizeof escaped, "\\u%04x", c);
            out += escaped;
        } else {
            out += c;
        }
    }
    return out;
}

std::string errorJson(std::string_view message)
{
    return "{\"error\":\"" + jsonEscape(message) + "\"}";
}

Analysis::Analysis(const std::string &networkPath)
    : _network(my_torch::Network::load(networkPath)),
      _encoding(analyzer::findEncoding(_network.info().encoding))
{
}

std::string Analysis::analyze(std::string_view fen)
{
    const analyzer::Position position = analyzer::parseFen(fen);
    my_torch::Matrix input(1, _encoding.width());
    _encoding.encode(position, &input(0, 0));
    const my_torch::Matrix output = _network.predict(input);
    const std::vector<std::string> &labels = _network.info().labels;
    const std::size_t predicted = my_torch::classesOf(output)[0];

    std::string json = "{\"fen\":\"" + jsonEscape(analyzer::formatFen(position)) +
                       "\",\"prediction\":\"" + jsonEscape(labels[predicted]) +
                       "\",\"probabilities\":[";
    for (std::size_t c = 0; c < labels.size(); c++) {
        char p[32];
        std::snprintf(p, sizeof p, "%.6f", output(0, c));
        json += std::string(c ? "," : "") + "{\"label\":\"" + jsonEscape(labels[c]) +
                "\",\"p\":" + p + "}";
    }
    json += "],\"oracle\":\"" + jsonEscape(oracle::verdictName(oracle::classify(position))) + "\"";
    // Squares a8 to h1, as in the FEN (ADR-0035).
    const SquareSaliency squares = saliency(_network, _encoding, position);
    for (const auto &[name, values] : {std::pair{"gradient", &squares.gradient},
                                       std::pair{"occlusion", &squares.occlusion}}) {
        json += std::string(",\"") + name + "\":[";
        for (std::size_t s = 0; s < 64; s++) {
            char v[32];
            std::snprintf(v, sizeof v, "%.6f", (*values)[s]);
            json += std::string(s ? "," : "") + v;
        }
        json += "]";
    }
    return json + "}";
}

} // namespace gui
