#ifndef CPP_API_PARITY_MACROS_H_
#define CPP_API_PARITY_MACROS_H_

// clang-format off
#  if defined(_WIN32)
#    if defined(cpp_api_parity_tests_EXPORTS)
#      define CPP_API_PARITY __declspec(dllexport)
#    else
#      define CPP_API_PARITY __declspec(dllimport)
#    endif
#  else
#    define CPP_API_PARITY
#  endif
// clang-format on

#endif // CPP_API_PARITY_MACROS_H_
