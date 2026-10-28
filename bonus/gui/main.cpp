// Chess GUI server (ADR-0029): serves the page of bonus/gui/web/ and analyses each position the
// page sends with the network and the rule-based oracle. Listens on 127.0.0.1 only, one request at
// a time.
//
//   my_torch_gui NETWORK [PORT (default 8080)] [WEB_DIRECTORY (default gui/web next to the binary)]
//
// Routes: GET /api/analyze?fen=FEN (JSON, see Analysis.hpp), GET /api/info (the network's file
// name, encoding and labels); any other GET path is a file of WEB_DIRECTORY ("/" is index.html).
// Errors at start-up go to stderr, exit code 84.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "Analysis.hpp"
#include "my_torch/Error.hpp"

namespace {

constexpr int EXIT_ERROR = 84;
constexpr std::size_t MAX_REQUEST = 16384;

struct Response {
    int status = 200;
    std::string type = "application/json";
    std::string body;
};

// Closes the descriptor when it goes out of scope.
class Socket {
  public:
    explicit Socket(int fd) : _fd(fd)
    {
    }
    ~Socket()
    {
        if (_fd >= 0)
            close(_fd);
    }
    Socket(const Socket &) = delete;
    Socket &operator=(const Socket &) = delete;
    int fd() const
    {
        return _fd;
    }

  private:
    int _fd;
};

std::string contentType(const std::filesystem::path &file)
{
    const std::string ext = file.extension().string();
    if (ext == ".html")
        return "text/html; charset=utf-8";
    if (ext == ".js")
        return "text/javascript; charset=utf-8";
    if (ext == ".css")
        return "text/css; charset=utf-8";
    if (ext == ".png")
        return "image/png";
    return "application/octet-stream";
}

// A file of the web directory. Only letters, digits, '-', '_', '.' and '/' are accepted, and no
// "..": a request cannot leave the directory.
Response staticFile(const std::filesystem::path &root, std::string path)
{
    if (path == "/")
        path = "/index.html";
    const bool safe = path.find("..") == std::string::npos &&
                      path.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                             "0123456789-_./") == std::string::npos;
    if (!safe)
        return {404, "text/plain", "not found"};
    const std::filesystem::path file = root / path.substr(1);
    std::ifstream in(file, std::ios::binary);
    if (!in || std::filesystem::is_directory(file))
        return {404, "text/plain", "not found"};
    std::stringstream body;
    body << in.rdbuf();
    return {200, contentType(file), body.str()};
}

Response handle(const std::string &request, gui::Analysis &analysis,
                const std::filesystem::path &root, const std::string &info)
{
    std::istringstream line(request.substr(0, request.find("\r\n")));
    std::string method;
    std::string target;
    line >> method >> target;
    if (method != "GET")
        return {405, "application/json", gui::errorJson("only GET is supported")};
    const std::size_t question = target.find('?');
    const std::string path = target.substr(0, question);
    const std::string query = question == std::string::npos ? "" : target.substr(question + 1);
    try {
        if (path == "/api/analyze")
            return {200, "application/json", analysis.analyze(gui::queryValue(query, "fen"))};
        if (path == "/api/info")
            return {200, "application/json", info};
    } catch (const std::exception &e) {
        return {400, "application/json", gui::errorJson(e.what())};
    }
    return staticFile(root, path);
}

const char *reason(int status)
{
    switch (status) {
    case 200:
        return "OK";
    case 400:
        return "Bad Request";
    case 404:
        return "Not Found";
    default:
        return "Method Not Allowed";
    }
}

void reply(int fd, const Response &r)
{
    const std::string all = "HTTP/1.1 " + std::to_string(r.status) + " " + reason(r.status) +
                            "\r\nContent-Type: " + r.type +
                            "\r\nContent-Length: " + std::to_string(r.body.size()) +
                            "\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n" + r.body;
    for (std::size_t sent = 0; sent < all.size();) {
        const ssize_t n = send(fd, all.data() + sent, all.size() - sent, MSG_NOSIGNAL);
        if (n <= 0)
            return;
        sent += static_cast<std::size_t>(n);
    }
}

std::string readRequest(int fd)
{
    std::string request;
    char buffer[4096];
    while (request.find("\r\n\r\n") == std::string::npos && request.size() < MAX_REQUEST) {
        const ssize_t n = recv(fd, buffer, sizeof buffer, 0);
        if (n <= 0)
            break;
        request.append(buffer, static_cast<std::size_t>(n));
    }
    return request;
}

std::string infoJson(const std::string &networkPath)
{
    const my_torch::Network net = my_torch::Network::load(networkPath);
    std::string json = "{\"network\":\"" +
                       gui::jsonEscape(std::filesystem::path(networkPath).filename().string()) +
                       "\",\"encoding\":\"" + gui::jsonEscape(net.info().encoding) +
                       "\",\"labels\":[";
    for (std::size_t i = 0; i < net.info().labels.size(); i++)
        json += std::string(i ? "," : "") + "\"" + gui::jsonEscape(net.info().labels[i]) + "\"";
    return json + "]}";
}

int port(const std::string &text)
{
    std::size_t used = 0;
    int value = 0;
    try {
        value = std::stoi(text, &used);
    } catch (const std::exception &) {
        used = 0;
    }
    if (used == 0 || used != text.size() || value < 1 || value > 65535)
        throw my_torch::Error("'" + text + "' is not a port number");
    return value;
}

void run(int argc, char **argv)
{
    if (argc < 2 || argc > 4)
        throw my_torch::Error("usage: my_torch_gui NETWORK [PORT] [WEB_DIRECTORY]");
    gui::Analysis analysis(argv[1]);
    const std::string info = infoJson(argv[1]);
    const int listenPort = argc > 2 ? port(argv[2]) : 8080;
    const std::filesystem::path root =
        argc > 3 ? std::filesystem::path(argv[3])
                 : std::filesystem::path(argv[0]).parent_path() / "gui" / "web";
    if (!std::filesystem::is_regular_file(root / "index.html"))
        throw my_torch::Error("no index.html in '" + root.string() + "'");

    const Socket server(socket(AF_INET, SOCK_STREAM, 0));
    if (server.fd() < 0)
        throw my_torch::Error("cannot create a socket");
    const int yes = 1;
    setsockopt(server.fd(), SOL_SOCKET, SO_REUSEADDR, &yes, sizeof yes);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<std::uint16_t>(listenPort));
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(server.fd(), reinterpret_cast<sockaddr *>(&address), sizeof address) < 0 ||
        listen(server.fd(), 16) < 0)
        throw my_torch::Error("cannot listen on 127.0.0.1:" + std::to_string(listenPort) +
                              " (port in use?)");
    std::cout << "my_torch_gui: open http://localhost:" << listenPort << " (Ctrl+C to stop)"
              << std::endl;
    while (true) {
        const Socket client(accept(server.fd(), nullptr, nullptr));
        if (client.fd() < 0)
            continue;
        // A client that connects and sends nothing must not block the server.
        const timeval timeout{2, 0};
        setsockopt(client.fd(), SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof timeout);
        reply(client.fd(), handle(readRequest(client.fd()), analysis, root, info));
    }
}

} // namespace

int main(int argc, char **argv)
{
    try {
        run(argc, argv);
    } catch (const std::exception &e) {
        std::cerr << "my_torch_gui: " << e.what() << '\n';
        return EXIT_ERROR;
    }
    return 0;
}
