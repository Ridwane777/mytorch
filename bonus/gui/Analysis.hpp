#pragma once

#include <string>
#include <string_view>

#include "Encoding.hpp"
#include "my_torch/Network.hpp"

// Back end of the chess GUI (ADR-0029): what the local server answers, kept apart from the sockets
// so that it can be tested.
namespace gui {

// Decodes a URL query component: %XX escapes and '+' for a space. Throws my_torch::Error on a
// malformed escape.
std::string urlDecode(std::string_view text);

// The decoded value of `key` in a query string ("fen=...&x=1"); empty if absent.
std::string queryValue(std::string_view query, std::string_view key);

// Escapes a string for a JSON string literal (quotes not included).
std::string jsonEscape(std::string_view text);

// {"error": message}
std::string errorJson(std::string_view message);

// Loads a network once and analyses positions with it and with the rule-based oracle.
class Analysis {
  public:
    explicit Analysis(const std::string &networkPath);

    // JSON for one FEN: the position as read back, the network's class and the probability of
    // each label, and the oracle's verdict. Throws my_torch::Error on an invalid FEN.
    std::string analyze(std::string_view fen);

  private:
    my_torch::Network _network;
    const analyzer::Encoding &_encoding;
};

} // namespace gui
