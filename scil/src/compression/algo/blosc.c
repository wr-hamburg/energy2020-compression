// This file is part of SCIL.
//
// SCIL is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// SCIL is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with SCIL.  If not, see <http://www.gnu.org/licenses/>.

#include <algo/blosc.h>

#include <string.h>
#include <blosc/blosc.h>

#pragma GCC diagnostic ignored "-Wunused-parameter"
int scil_blosc_compress(const scil_context_t* ctx, byte* restrict dest, size_t * restrict out_size, const byte*restrict source, const size_t source_size){
    int clevel = 5; /* Compression level default */
    int doshuffle = 1; /* Shuffle default */
    char* compname = "blosclz";
    //int code = BLOSC_BLOSCLZ;
    int type_size = sizeof(double);
    if(ctx->datatype == SCIL_TYPE_FLOAT){
        type_size = sizeof(float);
    }
    blosc_set_compressor(compname);
    *out_size = (size_t) blosc_compress(clevel, doshuffle, type_size, source_size,
                            source, dest, source_size + BLOSC_MAX_OVERHEAD);
    if (*out_size <= 0){
        printf("Blosc compression error");
        return -1;
    }
    return 0;
}

#pragma GCC diagnostic ignored "-Wunused-parameter"
int scil_blosc_decompress(byte*restrict dest, size_t buff_size, const byte*restrict src, const size_t in_size, size_t * uncomp_size_out){
    size_t size;
    size = *((size_t *) src);
    *uncomp_size_out = size;
    //size = ZSTD_decompress(dest, buff_size, src, in_size);
    //blosc_cbuffer_sizes(*buf, &outbuf_size, &cbytes, &blocksize);
    size = (size_t) blosc_decompress(src, dest, buff_size);
    if(size <= 0){
        return -1;
    }
    return 0;
}

scilU_algorithm_t algo_blosc = {
        .c.Btype = {
                scil_blosc_compress,
                scil_blosc_decompress
        },
        "blosc",
        19,
        SCIL_COMPRESSOR_TYPE_INDIVIDUAL_BYTES
};
