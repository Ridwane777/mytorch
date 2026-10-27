#include "my_torch/Parallel.hpp"

#include <algorithm>
#include <charconv>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace my_torch {

namespace {

std::size_t defaultCount()
{
    if (const char *env = std::getenv("MY_TORCH_THREADS")) {
        std::size_t value = 0;
        const auto [end, error] = std::from_chars(env, env + std::strlen(env), value);
        if (error == std::errc{} && *end == '\0' && value > 0)
            return value;
    }
    return std::max<std::size_t>(1, std::thread::hardware_concurrency());
}

// Workers wait for a job; the caller of run() works too and waits for the others to finish.
class Pool {
  public:
    explicit Pool(std::size_t threads)
    {
        for (std::size_t i = 1; i < threads; i++)
            _workers.emplace_back([this, i] { work(i); });
    }

    ~Pool()
    {
        {
            const std::lock_guard lock(_mutex);
            _stop = true;
        }
        _wake.notify_all();
        for (std::thread &t : _workers)
            t.join();
    }

    Pool(const Pool &) = delete;
    Pool &operator=(const Pool &) = delete;

    std::size_t size() const noexcept
    {
        return _workers.size() + 1;
    }

    // Splits [0, count) into size() contiguous ranges; range r goes to thread r.
    void run(std::size_t count, const std::function<void(std::size_t, std::size_t)> &body)
    {
        {
            const std::lock_guard lock(_mutex);
            _body = &body;
            _count = count;
            _pending = _workers.size();
            _generation++;
        }
        _wake.notify_all();
        runRange(0, body, count);
        std::unique_lock lock(_mutex);
        _done.wait(lock, [this] { return _pending == 0; });
        _body = nullptr;
    }

  private:
    void runRange(std::size_t index, const std::function<void(std::size_t, std::size_t)> &body,
                  std::size_t count) const
    {
        const std::size_t parts = size();
        const std::size_t begin = count * index / parts;
        const std::size_t end = count * (index + 1) / parts;
        if (begin < end)
            body(begin, end);
    }

    void work(std::size_t index)
    {
        std::size_t seen = 0;
        while (true) {
            const std::function<void(std::size_t, std::size_t)> *body = nullptr;
            std::size_t count = 0;
            {
                std::unique_lock lock(_mutex);
                _wake.wait(lock, [&] { return _stop || _generation != seen; });
                if (_stop)
                    return;
                seen = _generation;
                body = _body;
                count = _count;
            }
            runRange(index, *body, count);
            {
                const std::lock_guard lock(_mutex);
                if (--_pending == 0)
                    _done.notify_one();
            }
        }
    }

    std::vector<std::thread> _workers;
    std::mutex _mutex;
    std::condition_variable _wake;
    std::condition_variable _done;
    const std::function<void(std::size_t, std::size_t)> *_body = nullptr;
    std::size_t _count = 0;
    std::size_t _pending = 0;
    std::size_t _generation = 0;
    bool _stop = false;
};

std::mutex &poolMutex()
{
    static std::mutex mutex;
    return mutex;
}

std::unique_ptr<Pool> &pool()
{
    static std::unique_ptr<Pool> instance;
    return instance;
}

std::size_t &requested()
{
    static std::size_t count = defaultCount();
    return count;
}

} // namespace

std::size_t threadCount()
{
    const std::lock_guard lock(poolMutex());
    return requested();
}

void setThreadCount(std::size_t count)
{
    const std::lock_guard lock(poolMutex());
    requested() = count == 0 ? defaultCount() : count;
    pool().reset();
}

void parallelFor(std::size_t count, const std::function<void(std::size_t, std::size_t)> &body)
{
    if (count == 0)
        return;
    // One job at a time: a second caller waits (the library itself calls from one thread).
    const std::lock_guard lock(poolMutex());
    if (requested() == 1 || count == 1) {
        body(0, count);
        return;
    }
    if (!pool() || pool()->size() != requested())
        pool() = std::make_unique<Pool>(requested());
    pool()->run(count, body);
}

} // namespace my_torch
