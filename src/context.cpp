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

Context::Context(lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> ccParams, bool enable_fhe)
    : security_level_{ccParams.GetSecurityLevel()} {
    crypto_context_ = GenCryptoContext(ccParams);
    crypto_context_->Enable(lbcrypto::PKE);
    crypto_context_->Enable(lbcrypto::LEVELEDSHE);
    crypto_context_->Enable(lbcrypto::ADVANCEDSHE);
    crypto_context_->Enable(lbcrypto::KEYSWITCH);
    if (enable_fhe) {
        crypto_context_->Enable(lbcrypto::FHE);
    }
}

//////////////////////////////////////////////////////////////////////////////
// Key management

void Context::GenerateKeys() {
    key_pair_ = crypto_context_->KeyGen();
    crypto_context_->EvalMultKeyGen(key_pair_.secretKey);
}

void Context::GenerateSumKey() {
    if (key_pair_.secretKey == nullptr) {
        spdlog::error("Secret key is not set. Cannot generate sum key.");
        throw std::invalid_argument("Secret key is not set. Cannot generate sum key.");
    }

    crypto_context_->EvalSumKeyGen(key_pair_.secretKey);
}

void Context::GenerateRotateKeys(const std::vector<int>& indices) {
    if (key_pair_.secretKey == nullptr) {
        spdlog::error("Secret key is not set. Cannot generate rotation keys.");
        throw std::invalid_argument("Secret key is not set. Cannot generate rotation keys.");
    }

    crypto_context_->EvalRotateKeyGen(key_pair_.secretKey, indices);
}

void Context::GenerateBootstrapKeys() {
    if (key_pair_.secretKey == nullptr) {
        spdlog::error("Secret key is not set. Cannot generate bootstrapping keys.");
        throw std::invalid_argument("Secret key is not set. Cannot generate bootstrapping keys.");
    }

    crypto_context_->EvalBootstrapSetup({4, 4});
    crypto_context_->EvalBootstrapKeyGen(key_pair_.secretKey, crypto_context_->GetRingDimension() / 2);
}

bool Context::HasRotationIdx(int idx) const {
    auto cc      = GetCryptoContext();
    auto key_map = cc->GetEvalAutomorphismKeyMap(key_pair_.publicKey->GetKeyTag());
    auto am_idx  = lbcrypto::FindAutomorphismIndex2n(idx, cc->GetCyclotomicOrder());
    return key_map.count(am_idx) == 1;
}

//////////////////////////////////////////////////////////////////////////////
// File I/O

void Context::Load(const std::filesystem::path& path) {
    LoadCryptoContext(path / "cryptocontext.txt");
    LoadEvalMultKeys(path / "key-eval-mult.txt");

    if (std::filesystem::exists(path / "key-eval-sum.txt")) {
        LoadEvalSumKeys(path / "key-eval-sum.txt");
    }

    if (std::filesystem::exists(path / "key-rotate.txt")) {
        LoadRotationKeys(path / "key-rotate.txt");
    }
}

void Context::Save(const std::filesystem::path& path) const {
    if (!std::filesystem::exists(path)) {
        std::filesystem::create_directories(path);
    }

    SaveCryptoContext(path / "cryptocontext.txt");
    SaveEvalMultKeys(path / "key-eval-mult.txt");
    SaveEvalSumKeys(path / "key-eval-sum.txt");
    SaveRotationKeys(path / "key-rotate.txt");
}

void Context::SaveCryptoContext(const std::filesystem::path& path) const {
    if (!lbcrypto::Serial::SerializeToFile(path.string(), crypto_context_, lbcrypto::SerType::BINARY)) {
        throw std::filesystem::filesystem_error("Error writing serialization of the crypto context to " + path.string(),
                                                std::make_error_code(std::errc::io_error));
    }
}

void Context::LoadCryptoContext(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        spdlog::error("Invalid path to crypto context: {}", path.string());

        throw std::filesystem::filesystem_error("Directory does not exist.", std::make_error_code(std::errc::io_error));
    }

    if (!lbcrypto::Serial::DeserializeFromFile(path.string(), crypto_context_, lbcrypto::SerType::BINARY)) {
        spdlog::error("Error reading serialization of the crypto context.");

        throw std::filesystem::filesystem_error("Error reading serialization of the crypto context.",
                                                std::make_error_code(std::errc::io_error));
    }
}

void Context::LoadEvalMultKeys(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        spdlog::error("Error reading serialization of the eval mult keys.");

        throw std::filesystem::filesystem_error("Mult key file does not exist.",
                                                std::make_error_code(std::errc::io_error));
    }

    std::ifstream mult_key_istream{path, std::ios::in | std::ios::binary};
    if (!crypto_context_->DeserializeEvalMultKey(mult_key_istream, lbcrypto::SerType::BINARY)) {
        spdlog::error("Could not deserialize the eval mult key file");
        throw std::filesystem::filesystem_error("Could not deserialize the eval mult key file",
                                                std::make_error_code(std::errc::io_error));
    }
}

void Context::SaveEvalMultKeys(const std::filesystem::path& path) const {
    std::ofstream emkeyfile(path.string(), std::ios::out | std::ios::binary);
    if (emkeyfile.is_open()) {
        if (!crypto_context_->SerializeEvalMultKey(emkeyfile, lbcrypto::SerType::BINARY)) {
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

void Context::LoadEvalSumKeys(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        spdlog::error("Error reading serialization of the eval sum keys.");

        throw std::filesystem::filesystem_error("Sum key file does not exist.",
                                                std::make_error_code(std::errc::io_error));
    }

    std::ifstream sum_key_istream{path, std::ios::in | std::ios::binary};
    if (!crypto_context_->DeserializeEvalSumKey(sum_key_istream, lbcrypto::SerType::BINARY)) {
        spdlog::error("Could not deserialize the eval sum key file");
        throw std::filesystem::filesystem_error("Could not deserialize the eval sum key file",
                                                std::make_error_code(std::errc::io_error));
    }
}

void Context::SaveEvalSumKeys(const std::filesystem::path& path) const {
    std::ofstream sum_key_ostream(path.string(), std::ios::out | std::ios::binary);
    if (sum_key_ostream.is_open()) {
        if (!crypto_context_->SerializeEvalSumKey(sum_key_ostream, lbcrypto::SerType::BINARY)) {
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

void Context::LoadRotationKeys(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        spdlog::error("Error reading serialization of the eval rotate keys.");

        throw std::filesystem::filesystem_error("Rotate key file does not exist.",
                                                std::make_error_code(std::errc::io_error));
    }

    std::ifstream rotate_key_istream{path, std::ios::in | std::ios::binary};
    if (!crypto_context_->DeserializeEvalAutomorphismKey(rotate_key_istream, lbcrypto::SerType::BINARY)) {
        spdlog::error("Could not deserialize the eval rotate key file");
        throw std::filesystem::filesystem_error("Could not deserialize the eval rotate key file",
                                                std::make_error_code(std::errc::io_error));
    }
}

void Context::SaveRotationKeys(const std::filesystem::path& path) const {
    std::ofstream rotate_key_file(path.string(), std::ios::out | std::ios::binary);

    if (rotate_key_file.is_open()) {
        if (!crypto_context_->SerializeEvalAutomorphismKey(rotate_key_file, lbcrypto::SerType::BINARY)) {
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

void Context::SavePublicKey(const std::filesystem::path& path) const {
    if (!lbcrypto::Serial::SerializeToFile(path.string(), key_pair_.publicKey, lbcrypto::SerType::BINARY)) {
        spdlog::error("Error writing serialization of public key");
        throw std::filesystem::filesystem_error("Error writing serialization of public key",
                                                std::make_error_code(std::errc::io_error));
    }
    spdlog::debug("The public key has been serialized.");
}

void Context::LoadPublicKey(const std::filesystem::path& path) {
    if (!lbcrypto::Serial::DeserializeFromFile(path.string(), key_pair_.publicKey, lbcrypto::SerType::BINARY)) {
        spdlog::error("Error reading serialization of public key");
        throw std::filesystem::filesystem_error("Error reading serialization of public key",
                                                std::make_error_code(std::errc::io_error));
    }
}

void Context::SaveSecretKey(const std::filesystem::path& path) const {
    if (!lbcrypto::Serial::SerializeToFile(path.string(), key_pair_.secretKey, lbcrypto::SerType::BINARY)) {
        spdlog::error("Error writing serialization of secret key");
        throw std::filesystem::filesystem_error("Error writing serialization of secret key",
                                                std::make_error_code(std::errc::io_error));
    }
    spdlog::debug("The secret key has been serialized.");
}

void Context::LoadSecretKey(const std::filesystem::path& path) {
    if (!lbcrypto::Serial::DeserializeFromFile(path.string(), key_pair_.secretKey, lbcrypto::SerType::BINARY)) {
        spdlog::error("Error reading serialization of secret key");
        throw std::filesystem::filesystem_error("Error reading serialization of secret key",
                                                std::make_error_code(std::errc::io_error));
    }
}
