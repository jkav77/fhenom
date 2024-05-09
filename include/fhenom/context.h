#pragma once

#include <openfhe.h>

#include <filesystem>

namespace fhenom {
class Context {
protected:
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> crypto_context_;
    lbcrypto::KeyPair<lbcrypto::DCRTPoly> key_pair_;
    lbcrypto::SecurityLevel security_level_{lbcrypto::HEStd_NotSet};

public:
    Context() = default;
    Context(lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> ccParams, bool enable_fhe = false);
    Context(std::filesystem::path savedContextPath) {
        Load(savedContextPath);
    }

    //////////////////////////////////////////////////////////////////////////////
    // Key Generation

    void GenerateKeys();
    void GenerateSumKey();
    void GenerateRotateKeys(const std::vector<int>& indices);
    void GenerateBootstrapKeys();

    //////////////////////////////////////////////////////////////////////////////
    // Getters and Setters

    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& GetCryptoContext() const {
        return crypto_context_;
    }

    void SetCryptoContext(const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context) {
        crypto_context_ = context;
    }

    const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& GetKeyPair() const {
        return key_pair_;
    }

    void SetKeyPair(const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys) {
        key_pair_ = keys;
    }

    bool HasRotationIdx(int idx) const;

    //////////////////////////////////////////////////////////////////////////////
    // File I/O
    /**
   * @brief Load crypto context and evaluation keys from file
   *
   * @param path The directory containing the crypto context and keys
   *
   * @note The crypto context and keys are loaded from the following files:
   * cryptocontext.txt, key-eval-mult.txt, key-eval-sum.txt
   */
    void Load(const std::filesystem::path& path);

    /**
   * @brief Save crypto context and evaluation keys to file
   *
   * @param path The destination directory
   *
   * @note The crypto context and keys are saved to the following files:
   * cryptocontext.txt, key-eval-mult.txt, key-eval-sum.txt
   */
    void Save(const std::filesystem::path& path) const;

    void LoadCryptoContext(const std::filesystem::path& path);
    void SaveCryptoContext(const std::filesystem::path& path) const;
    void LoadEvalMultKeys(const std::filesystem::path& path);
    void SaveEvalMultKeys(const std::filesystem::path& path) const;
    void LoadEvalSumKeys(const std::filesystem::path& path);
    void SaveEvalSumKeys(const std::filesystem::path& path) const;
    void LoadRotationKeys(const std::filesystem::path& path);
    void SaveRotationKeys(const std::filesystem::path& path) const;
    void LoadPublicKey(const std::filesystem::path& path);
    void SavePublicKey(const std::filesystem::path& path) const;
    void LoadSecretKey(const std::filesystem::path& path);
    void SaveSecretKey(const std::filesystem::path& path) const;
};
}  // namespace fhenom
