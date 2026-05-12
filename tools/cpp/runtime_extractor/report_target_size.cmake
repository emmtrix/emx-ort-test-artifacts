if(NOT DEFINED target_name OR NOT DEFINED target_file)
  message(FATAL_ERROR "target_name and target_file must be defined.")
endif()

if(NOT EXISTS "${target_file}")
  message(FATAL_ERROR "Target file does not exist: ${target_file}")
endif()

file(SIZE "${target_file}" target_size_bytes)
message(STATUS "Built ${target_name}: ${target_size_bytes} bytes (${target_file})")
