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
  Context(std::filesystem::path savedContextPath) { load(savedContextPath); }

  //////////////////////////////////////////////////////////////////////////////
  // Key Generation

  void generateKeys();
  void generateSumKey();
  void generateRotateKeys(std::vector<int> indices);

  //////////////////////////////////////////////////////////////////////////////
  // Getters and Setters

  const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& getCryptoContext() const {
    return cryptoContext;
  }

  void setCryptoContext(
      const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context) {
    cryptoContext = context;
  }

  const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& getKeyPair() const {
    return keyPair;
  }

  void setKeyPair(const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys) {
    keyPair = keys;
  }

  bool hasRotationIdx(int idx) const;

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
  void load(const std::filesystem::path path);

  /**
   * @brief Save crypto context and evaluation keys to file
   *
   * @param path The destination directory
   *
   * @note The crypto context and keys are saved to the following files:
   * cryptocontext.txt, key-eval-mult.txt, key-eval-sum.txt
   */
  void save(const std::filesystem::path path);

  void loadCryptoContext(const std::filesystem::path path);
  void saveCryptoContext(const std::filesystem::path path);
  void loadEvalMultKeys(const std::filesystem::path path);
  void saveEvalMultKeys(const std::filesystem::path path);
  void loadEvalSumKeys(const std::filesystem::path path);
  void saveEvalSumKeys(const std::filesystem::path path);
  void loadRotationKeys(const std::filesystem::path path);
  void saveRotationKeys(const std::filesystem::path path);
  void loadPublicKey(const std::filesystem::path path);
  void savePublicKey(const std::filesystem::path path);
  void loadSecretKey(const std::filesystem::path path);
  void saveSecretKey(const std::filesystem::path path);
};
}  // namespace fhenom
