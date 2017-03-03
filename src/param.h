#ifndef	_PARAMS_H_
#define	_PARAMS_H_

#define BLOCK_SIZE				8
#define BLOCK_SIZE_SQR				64
#define VARIANCE_INTERPOLATE_PARAM	        (1.0)

#define MAX_BG_AGE				30
#define VAR_MIN_NOISE_T			        2500.0f
#define VAR_DEC_RATIO			        (0.001)
#define MIN_BG_VAR				25.0f
#define INIT_BG_VAR				400.0f

#define NUM_MODELS		        (2)
#define VAR_THRESH_FG_DETERMINE		8.0  //4.0
#define VAR_THRESH_MODEL_MATCH		2.0
#endif				// _PARAMS_H_
