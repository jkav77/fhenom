#include <fhenom/context.h>

fhenom::Context GetFheContext() {
    lbcrypto::ScalingTechnique fhe_sc_tech = lbcrypto::FLEXIBLEAUTO;
    uint32_t fhe_scale_mod_size            = 59;
    uint32_t fhe_first_mod_size            = 60;
    lbcrypto::SecurityLevel fhe_sl         = lbcrypto::HEStd_128_classic;

    lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> fhe_ckks_parameters;
    uint32_t levels_available_after_bootstrap = 10;
    usint boostrap_depth = lbcrypto::FHECKKSRNS::GetBootstrapDepth({4, 4}, lbcrypto::UNIFORM_TERNARY);
    usint depth          = levels_available_after_bootstrap + boostrap_depth;
    fhe_ckks_parameters.SetMultiplicativeDepth(depth);
    fhe_ckks_parameters.SetScalingModSize(fhe_scale_mod_size);
    fhe_ckks_parameters.SetFirstModSize(fhe_first_mod_size);
    fhe_ckks_parameters.SetScalingTechnique(fhe_sc_tech);
    fhe_ckks_parameters.SetSecurityLevel(fhe_sl);
    fhe_ckks_parameters.SetRingDim(131072);
    fhe_ckks_parameters.SetSecretKeyDist(lbcrypto::UNIFORM_TERNARY);
    fhe_ckks_parameters.SetKeySwitchTechnique(lbcrypto::HYBRID);

    fhenom::Context fhe_context(fhe_ckks_parameters, true);

    return fhe_context;
}

fhenom::Context GetLeveledContext() {
    // ScalingTechnique sc_tech = FIXEDAUTO;
    lbcrypto::ScalingTechnique sc_tech = lbcrypto::FLEXIBLEAUTO;
    uint32_t mult_depth                = 2;
    if (sc_tech == lbcrypto::FLEXIBLEAUTOEXT)
        mult_depth += 1;
    uint32_t scale_mod_size    = 24;
    uint32_t first_mod_size    = 30;
    uint32_t ring_dim          = 8192;
    lbcrypto::SecurityLevel sl = lbcrypto::HEStd_128_classic;

    lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> ckks_parameters;
    ckks_parameters.SetMultiplicativeDepth(mult_depth);
    ckks_parameters.SetScalingModSize(scale_mod_size);
    ckks_parameters.SetFirstModSize(first_mod_size);
    ckks_parameters.SetScalingTechnique(sc_tech);
    ckks_parameters.SetSecurityLevel(sl);
    ckks_parameters.SetRingDim(ring_dim);
    ckks_parameters.SetSecretKeyDist(lbcrypto::UNIFORM_TERNARY);
    ckks_parameters.SetKeySwitchTechnique(lbcrypto::HYBRID);
    ckks_parameters.SetNumLargeDigits(3);

    fhenom::Context context{ckks_parameters};

    return context;
}
