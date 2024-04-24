#include <fhenom/context.h>
#include <openfhe.h>
#include <spdlog/spdlog.h>

#include <filesystem>

// header files needed for serialization
#include <ciphertext-ser.h>
#include <cryptocontext-ser.h>
#include <key/key-ser.h>
#include <scheme/ckksrns/ckksrns-ser.h>

using fhenom::Context;

Context::Context(lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> ccParams)
    : ckksSecurityLevel{ccParams.GetSecurityLevel()} {
    cryptoContext = GenCryptoContext(ccParams);
    cryptoContext->Enable(lbcrypto::PKE);
    cryptoContext->Enable(lbcrypto::LEVELEDSHE);
    cryptoContext->Enable(lbcrypto::ADVANCEDSHE);
    cryptoContext->Enable(lbcrypto::KEYSWITCH);
}

//////////////////////////////////////////////////////////////////////////////
// Key management

void Context::generateKeys() {
    keyPair = cryptoContext->KeyGen();
    cryptoContext->EvalMultKeyGen(keyPair.secretKey);
}

void Context::generateSumKey() {
    if (keyPair.secretKey == nullptr) {
        spdlog::error("Secret key is not set. Cannot generate sum key.");
        throw std::invalid_argument("Secret key is not set. Cannot generate sum key.");
    }

    cryptoContext->EvalSumKeyGen(keyPair.secretKey);
}

void Context::generateRotateKeys(const std::vector<int>& indices) {
    if (keyPair.secretKey == nullptr) {
        spdlog::error("Secret key is not set. Cannot generate rotation keys.");
        throw std::invalid_argument("Secret key is not set. Cannot generate rotation keys.");
    }

    cryptoContext->EvalRotateKeyGen(keyPair.secretKey, indices);
}

bool Context::hasRotationIdx(int idx) const {
    auto cc      = getCryptoContext();
    auto key_map = cc->GetEvalAutomorphismKeyMap(keyPair.publicKey->GetKeyTag());
    auto am_idx  = lbcrypto::FindAutomorphismIndex2n(idx, cc->GetCyclotomicOrder());
    return key_map.count(am_idx) == 1;
}

//////////////////////////////////////////////////////////////////////////////
// File I/O

void Context::load(const std::filesystem::path& path) {
    loadCryptoContext(path / "cryptocontext.txt");
    loadEvalMultKeys(path / "key-eval-mult.txt");

    if (std::filesystem::exists(path / "key-eval-sum.txt")) {
        loadEvalSumKeys(path / "key-eval-sum.txt");
    }

    if (std::filesystem::exists(path / "key-rotate.txt")) {
        loadRotationKeys(path / "key-rotate.txt");
    }
}

void Context::save(const std::filesystem::path& path) const {
    if (!std::filesystem::exists(path)) {
        std::filesystem::create_directories(path);
    }

    saveCryptoContext(path / "cryptocontext.txt");
    saveEvalMultKeys(path / "key-eval-mult.txt");
    saveEvalSumKeys(path / "key-eval-sum.txt");
    saveRotationKeys(path / "key-rotate.txt");
}

void Context::saveCryptoContext(const std::filesystem::path& path) const {
    if (!lbcrypto::Serial::SerializeToFile(path.string(), cryptoContext, lbcrypto::SerType::BINARY)) {
        throw std::filesystem::filesystem_error("Error writing serialization of the crypto context to " + path.string(),
                                                std::make_error_code(std::errc::io_error));
    }
}

void Context::loadCryptoContext(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        spdlog::error("Invalid path to crypto context: {}", path.string());

        throw std::filesystem::filesystem_error("Directory does not exist.", std::make_error_code(std::errc::io_error));
    }

    if (!lbcrypto::Serial::DeserializeFromFile(path.string(), cryptoContext, lbcrypto::SerType::BINARY)) {
        spdlog::error("Error reading serialization of the crypto context.");

        throw std::filesystem::filesystem_error("Error reading serialization of the crypto context.",
                                                std::make_error_code(std::errc::io_error));
    }
}

void Context::loadEvalMultKeys(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        spdlog::error("Error reading serialization of the eval mult keys.");

        throw std::filesystem::filesystem_error("Mult key file does not exist.",
                                                std::make_error_code(std::errc::io_error));
    }

    std::ifstream mult_key_istream{path, std::ios::in | std::ios::binary};
    if (!cryptoContext->DeserializeEvalMultKey(mult_key_istream, lbcrypto::SerType::BINARY)) {
        spdlog::error("Could not deserialize the eval mult key file");
        throw std::filesystem::filesystem_error("Could not deserialize the eval mult key file",
                                                std::make_error_code(std::errc::io_error));
    }
}

void Context::saveEvalMultKeys(const std::filesystem::path& path) const {
    std::ofstream emkeyfile(path.string(), std::ios::out | std::ios::binary);
    if (emkeyfile.is_open()) {
        if (!cryptoContext->SerializeEvalMultKey(emkeyfile, lbcrypto::SerType::BINARY)) {
            spdlog::error("Error writing serialization of the eval mult keys");
            throw std::filesystem::filesystem_error("Error writing serialization of the eval mult keys",
                                                    std::make_error_code(std::errc::io_error));
        }
        spdlog::debug("The eval mult keys have been serialized.");
        emkeyfile.close();
    }
    else {
        spdlog::error("Error opening eval mult key file for writing");
        throw std::filesystem::filesystem_error("Error opening key-eval-mult.txt for writing",
                                                std::make_error_code(std::errc::io_error));
    }
}

void Context::loadEvalSumKeys(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        spdlog::error("Error reading serialization of the eval sum keys.");

        throw std::filesystem::filesystem_error("Sum key file does not exist.",
                                                std::make_error_code(std::errc::io_error));
    }

    std::ifstream sum_key_istream{path, std::ios::in | std::ios::binary};
    if (!cryptoContext->DeserializeEvalSumKey(sum_key_istream, lbcrypto::SerType::BINARY)) {
        spdlog::error("Could not deserialize the eval sum key file");
        throw std::filesystem::filesystem_error("Could not deserialize the eval sum key file",
                                                std::make_error_code(std::errc::io_error));
    }
}

void Context::saveEvalSumKeys(const std::filesystem::path& path) const {
    std::ofstream sum_key_ostream(path.string(), std::ios::out | std::ios::binary);
    if (sum_key_ostream.is_open()) {
        if (!cryptoContext->SerializeEvalSumKey(sum_key_ostream, lbcrypto::SerType::BINARY)) {
            spdlog::error("Error writing serialization of the eval sum keys");
            throw std::filesystem::filesystem_error("Error writing serialization of the eval sum keys",
                                                    std::make_error_code(std::errc::io_error));
        }
        spdlog::debug("The eval sum keys have been serialized.");
        sum_key_ostream.close();
    }
    else {
        spdlog::error("Error opening eval sum key file for writing");
        throw std::filesystem::filesystem_error("Error opening key-eval-sum.txt for writing",
                                                std::make_error_code(std::errc::io_error));
    }
}

void Context::loadRotationKeys(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        spdlog::error("Error reading serialization of the eval rotate keys.");

        throw std::filesystem::filesystem_error("Rotate key file does not exist.",
                                                std::make_error_code(std::errc::io_error));
    }

    std::ifstream rotate_key_istream{path, std::ios::in | std::ios::binary};
    if (!cryptoContext->DeserializeEvalAutomorphismKey(rotate_key_istream, lbcrypto::SerType::BINARY)) {
        spdlog::error("Could not deserialize the eval rotate key file");
        throw std::filesystem::filesystem_error("Could not deserialize the eval rotate key file",
                                                std::make_error_code(std::errc::io_error));
    }
}

void Context::saveRotationKeys(const std::filesystem::path& path) const {
    std::ofstream rotate_key_file(path.string(), std::ios::out | std::ios::binary);

    if (rotate_key_file.is_open()) {
        if (!cryptoContext->SerializeEvalAutomorphismKey(rotate_key_file, lbcrypto::SerType::BINARY)) {
            spdlog::error("Error writing serialization of the eval rotate keys");
            throw std::filesystem::filesystem_error("Error writing serialization of the eval rotate keys",
                                                    std::make_error_code(std::errc::io_error));
        }
        spdlog::debug("The eval rotate keys have been serialized.");
        rotate_key_file.close();
    }
    else {
        spdlog::error("Error opening eval rotate key file for writing");
        throw std::filesystem::filesystem_error("Error opening key-rotate.txt for writing",
                                                std::make_error_code(std::errc::io_error));
    }
}

void Context::savePublicKey(const std::filesystem::path& path) const {
    if (!lbcrypto::Serial::SerializeToFile(path.string(), keyPair.publicKey, lbcrypto::SerType::BINARY)) {
        spdlog::error("Error writing serialization of public key");
        throw std::filesystem::filesystem_error("Error writing serialization of public key",
                                                std::make_error_code(std::errc::io_error));
    }
    spdlog::debug("The public key has been serialized.");
}

void Context::loadPublicKey(const std::filesystem::path& path) {
    if (!lbcrypto::Serial::DeserializeFromFile(path.string(), keyPair.publicKey, lbcrypto::SerType::BINARY)) {
        spdlog::error("Error reading serialization of public key");
        throw std::filesystem::filesystem_error("Error reading serialization of public key",
                                                std::make_error_code(std::errc::io_error));
    }
}

void Context::saveSecretKey(const std::filesystem::path& path) const {
    if (!lbcrypto::Serial::SerializeToFile(path.string(), keyPair.secretKey, lbcrypto::SerType::BINARY)) {
        spdlog::error("Error writing serialization of secret key");
        throw std::filesystem::filesystem_error("Error writing serialization of secret key",
                                                std::make_error_code(std::errc::io_error));
    }
    spdlog::debug("The secret key has been serialized.");
}

void Context::loadSecretKey(const std::filesystem::path& path) {
    if (!lbcrypto::Serial::DeserializeFromFile(path.string(), keyPair.secretKey, lbcrypto::SerType::BINARY)) {
        spdlog::error("Error reading serialization of secret key");
        throw std::filesystem::filesystem_error("Error reading serialization of secret key",
                                                std::make_error_code(std::errc::io_error));
    }
}
