#ifndef FAST_WRAPPER_H
#define FAST_WRAPPER_H

namespace exw_fast {
#ifdef AMR_WIND_USE_OPENFAST
extern "C" {
#include "FAST_Library.h"
}

extern void exw_fast_output_redirect(char*);

inline constexpr int fast_strlen() { return INTERFACE_STRING_LENGTH; }

#else

#define ErrID_None 0
#define ErrID_Info 1
#define ErrID_Warn 2
#define ErrID_Severe 3
#define ErrID_Fatal 4

struct ExtInfw_InputType_t
{};
struct ExtInfw_OutputType_t
{};

#ifdef AMR_WIND_FAST_USE_SCDX
struct SC_DX_InputType
{};
struct SC_DX_OutputType
{};
#else
struct SC_InputType
{};
struct SC_OutputType
{};
#endif

inline constexpr int fast_strlen() { return 1025; }

inline void exw_fast_output_redirect(char* /*unused*/) {}

inline void
FAST_AllocateTurbines(int* /*unused*/, int* /*unused*/, char* /*unused*/)
{}
inline void FAST_DeallocateTurbines(int* /*unused*/, char* /*unused*/) {}
inline void
FAST_CFD_Solution0(int* /*unused*/, int* /*unused*/, char* /*unused*/)
{}
inline void FAST_CFD_Step(int* /*unused*/, int* /*unused*/, char* /*unused*/) {}
inline void FAST_CreateCheckpoint(
    int* /*unused*/, char* /*unused*/, int* /*unused*/, char* /*unused*/)
{}

// clang-format off
#ifdef AMR_WIND_FAST_USE_SCDX
inline void FAST_ExtInfw_Init(int *  , double * , const char * , int *  , char * ,
                                        int *  , int *  , int *  , float *  , float *  ,
                                        int *  , int *  , float *  , int * ,
                                        double *  , double *  , int *  , int *  , int *  , int *  , int *  ,
                                        ExtInfw_InputType_t*  , ExtInfw_OutputType_t*  , SC_InputType*  , SC_OutputType*  , 
                                        int * , char * ) {}
inline void FAST_ExtInfw_Restart(
    int* , char* , int* , double* , int* , int* , int* , int* , int* ,
    ExtInfw_InputType_t* , ExtInfw_OutputType_t* ,
    SC_InputType* , SC_OutputType* , int* , char* ) {}
#else
inline void FAST_ExtInfw_Init(int * /*unused*/ , double * /*unused*/, const char * /*unused*/, int * /*unused*/ , char * /*unused*/,
                                        int * /*unused*/ , int * /*unused*/ , int * /*unused*/ , float * /*unused*/ , float * /*unused*/ ,
                                        int * /*unused*/ , int * /*unused*/ , float * /*unused*/ , int * /*unused*/,
                                        double * /*unused*/ , double * /*unused*/ , int * /*unused*/ , int * /*unused*/ , int * /*unused*/ , int * /*unused*/ , int * /*unused*/ ,
                                        ExtInfw_InputType_t* /*unused*/ , ExtInfw_OutputType_t* /*unused*/ , SC_InputType* /*unused*/ , SC_OutputType* /*unused*/ , 
                                        int * /*unused*/, char * /*unused*/) {}

inline void FAST_ExtInfw_Restart(
    int* /*unused*/, char* /*unused*/, int* /*unused*/, double* /*unused*/, int* /*unused*/, int* /*unused*/, int* /*unused*/, int* /*unused*/, int* /*unused*/,
    ExtInfw_InputType_t* /*unused*/, ExtInfw_OutputType_t* /*unused*/,
    SC_InputType* /*unused*/, SC_OutputType* /*unused*/, int* /*unused*/, char* /*unused*/) {}
#endif
// clang-format on
#endif
} // namespace exw_fast

#endif /* FAST_WRAPPER_H */
