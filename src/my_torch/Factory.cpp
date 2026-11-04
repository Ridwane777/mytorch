#include "my_torch/Factory.hpp"

#include <charconv>
#include <cmath>

#include "my_torch/Error.hpp"
#include "my_torch/Losses.hpp"
#include "my_torch/Optimizers.hpp"

namespace my_torch {

namespace {

// Reads the settings of one component and refuses the keys nobody asked for.
class Settings {
  public:
    explicit Settings(const ComponentSpec &spec) : _spec(spec), _used(spec.params.size(), false)
    {
    }

    Scalar number(const std::string &key, Scalar fallback)
    {
        const std::string *text = find(key);
        return text ? parseNumber(key, *text) : fallback;
    }

    std::size_t count(const std::string &key, std::size_t fallback)
    {
        const std::string *text = find(key);
        if (!text)
            return fallback;
        std::size_t value = 0;
        const auto [end, error] = std::from_chars(text->data(), text->data() + text->size(), value);
        if (error != std::errc{} || end != text->data() + text->size())
            fail(key, *text, "a non-negative integer");
        return value;
    }

    bool flag(const std::string &key, bool fallback)
    {
        const std::string *text = find(key);
        if (!text)
            return fallback;
        if (*text != "0" && *text != "1")
            fail(key, *text, "0 or 1");
        return *text == "1";
    }

    std::vector<Scalar> list(const std::string &key)
    {
        const std::string *text = find(key);
        std::vector<Scalar> values;
        if (!text)
            return values;
        std::size_t start = 0;
        while (true) {
            const std::size_t comma = text->find(',', start);
            values.push_back(parseNumber(key, text->substr(start, comma - start)));
            if (comma == std::string::npos)
                return values;
            start = comma + 1;
        }
    }

    // Call once every expected key has been read.
    void finish() const
    {
        for (std::size_t i = 0; i < _used.size(); i++)
            if (!_used[i])
                throw Error(_spec.type + ": unknown setting '" + _spec.params[i].first + "'");
    }

  private:
    const std::string *find(const std::string &key)
    {
        for (std::size_t i = 0; i < _spec.params.size(); i++)
            if (_spec.params[i].first == key) {
                _used[i] = true;
                return &_spec.params[i].second;
            }
        return nullptr;
    }

    Scalar parseNumber(const std::string &key, const std::string &text) const
    {
        Scalar value = 0;
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
        if (error != std::errc{} || end != text.data() + text.size() || !std::isfinite(value))
            fail(key, text, "a finite number");
        return value;
    }

    [[noreturn]] void fail(const std::string &key, const std::string &text, const char *what) const
    {
        throw Error(_spec.type + ": " + key + " must be " + what + ", got '" + text + "'");
    }

    const ComponentSpec &_spec;
    std::vector<bool> _used;
};

} // namespace

std::unique_ptr<Loss> makeLoss(const ComponentSpec &spec)
{
    Settings s(spec);
    std::unique_ptr<Loss> loss;

    if (spec.type == "mse")
        loss = std::make_unique<MSE>();
    else if (spec.type == "mae")
        loss = std::make_unique<MAE>();
    else if (spec.type == "huber")
        loss = std::make_unique<Huber>(s.number("delta", 1.0));
    else if (spec.type == "binary_cross_entropy")
        loss = std::make_unique<BinaryCrossEntropy>();
    else if (spec.type == "cross_entropy") {
        const Scalar smoothing = s.number("smoothing", 0);
        loss = std::make_unique<CrossEntropy>(smoothing, s.list("weights"));
    } else
        throw Error("unknown loss '" + spec.type + "'");
    s.finish();
    return loss;
}

std::unique_ptr<Optimizer> makeOptimizer(const ComponentSpec &spec)
{
    Settings s(spec);
    std::unique_ptr<Optimizer> optimizer;

    if (spec.type == "sgd") {
        const Scalar lr = s.number("lr", 0.01);
        const Scalar momentum = s.number("momentum", 0);
        optimizer = std::make_unique<SGD>(lr, momentum, s.flag("nesterov", false));
    } else if (spec.type == "adam" || spec.type == "adamw") {
        const Scalar lr = s.number("lr", 0.001);
        const Scalar decay = spec.type == "adamw" ? s.number("weight_decay", 0.01) : 0;
        const Scalar beta1 = s.number("beta1", 0.9);
        const Scalar beta2 = s.number("beta2", 0.999);
        const Scalar epsilon = s.number("epsilon", 1e-8);
        const bool amsgrad = s.flag("amsgrad", false);
        if (spec.type == "adam")
            optimizer = std::make_unique<Adam>(lr, beta1, beta2, epsilon, amsgrad);
        else
            optimizer = std::make_unique<AdamW>(lr, decay, beta1, beta2, epsilon, amsgrad);
    } else if (spec.type == "rmsprop") {
        const Scalar lr = s.number("lr", 0.001);
        const Scalar decay = s.number("decay", 0.9);
        optimizer = std::make_unique<RMSProp>(lr, decay, s.number("epsilon", 1e-8));
    } else if (spec.type == "adagrad") {
        const Scalar lr = s.number("lr", 0.01);
        optimizer = std::make_unique<Adagrad>(lr, s.number("epsilon", 1e-10));
    } else
        throw Error("unknown optimizer '" + spec.type + "'");
    s.finish();
    return optimizer;
}

std::unique_ptr<Scheduler> makeScheduler(const ComponentSpec &spec)
{
    if (spec.type == "none") {
        if (!spec.params.empty())
            throw Error("none: unknown setting '" + spec.params.front().first + "'");
        return nullptr;
    }
    Settings s(spec);
    std::unique_ptr<Scheduler> scheduler;

    if (spec.type == "constant")
        scheduler = std::make_unique<ConstantRate>(s.number("base", 0.001));
    else if (spec.type == "step") {
        const Scalar base = s.number("base", 0.001);
        const std::size_t step = s.count("step", 10);
        scheduler = std::make_unique<StepDecay>(base, step, s.number("gamma", 0.1));
    } else if (spec.type == "exponential") {
        const Scalar base = s.number("base", 0.001);
        scheduler = std::make_unique<ExponentialDecay>(base, s.number("gamma", 0.95));
    } else if (spec.type == "cosine") {
        const Scalar base = s.number("base", 0.001);
        const std::size_t epochs = s.count("epochs", 10);
        scheduler = std::make_unique<CosineAnnealing>(base, epochs, s.number("min", 0));
    } else
        throw Error("unknown scheduler '" + spec.type + "'");
    const std::size_t warmup = s.count("warmup", 0);
    s.finish();
    if (warmup > 0)
        scheduler = std::make_unique<Warmup>(std::move(scheduler), warmup);
    return scheduler;
}

} // namespace my_torch
