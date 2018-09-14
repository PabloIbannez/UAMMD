/*Raul P. Pelaez 2018
A simple function that uses libpng to write a png image from a pixel buffer.
Files including this header must also compile RPNG.c and use -lpng when linking.
*/
#ifndef RPNG_H
#define RPNG_H
#include<png.h>
#include<stdlib.h>
#include<string.h>


//Writes a png from a buffer given in px, the buffer size must be of size wx*wy*size_of(color format given in ctype)
//For example if ctype is the default RGBA px must have 4 uchars per pixel stored in px[i,j] = px[i+wx*j] layout.
bool savePNG(const char* fileName,
	     unsigned char* px, int wx, int wy,
	     int ctype = PNG_COLOR_TYPE_RGB_ALPHA);

#endif
