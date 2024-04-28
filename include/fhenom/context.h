#pragma once

#include <openfhe.h>

#include <filesystem>

namespace fhenom {
class Context {
protected:
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cryptoContext;
    lbcrypto::KeyPair<lbcrypto::DCRTPoly> keyPair;
    lbcrypto::SecurityLevel ckksSecurityLevel{lbcrypto::HEStd_NotSet};

public:
    Context() = default;
    Context(lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> ccParams);
    Context(std::filesystem::path savedContextPath) {
        Load(savedContextPath);
    }

    //////////////////////////////////////////////////////////////////////////////
    // Key Generation

    void GenerateKeys();
    void GenerateSumKey();
    void GenerateRotateKeys(const std::vector<int>& indices);

    //////////////////////////////////////////////////////////////////////////////
    // Getters and Setters

    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& GetCryptoContext() const {
        return cryptoContext;
    }

    void SetCryptoContext(const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context) {
        cryptoContext = context;
    }

    const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& GetKeyPair() const {
        return keyPair;
    }

    void SetKeyPair(const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys) {
        keyPair = keys;
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
