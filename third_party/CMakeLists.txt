set (THIRD_PARTY_DIR ${CMAKE_SOURCE_DIR}/third_party)

#############################
## Google Test
#############################

set (GTEST_HOME ${THIRD_PARTY_DIR}/googletest)

add_subdirectory(googletest)

set (GTEST_HOME ${GTEST_HOME} PARENT_SCOPE)
set (GTEST_BOTH_LIBRARIES gtest gtest_main PARENT_SCOPE)
set (THIRD_PARTY_DIR ${THIRD_PARTY_DIR} PARENT_SCOPE)
