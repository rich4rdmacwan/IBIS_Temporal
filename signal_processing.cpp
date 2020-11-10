#include "signal_processing.h"

#define VISU_SIGNAL_FINAL 0

Signal_processing::Signal_processing( int MaxSP, int size_signal)
{
    size_signals = size_signal;
    max_SP = MaxSP;
    index_circular = 0;
    HR = 0;
    count_SNR = 0;
    ready=false;
    complete_SNR=false;
    index_processed=0;

    buff_signals = new float[ max_SP * size_signals ];
    buff_signals_c1 = new float[ max_SP * size_signals ];
    buff_signals_c2 = new float[ max_SP * size_signals ];
    buff_signals_c3 = new float[ max_SP * size_signals ];
    circular_data_c1 = new float[ max_SP * size_signals ];
    circular_data_c2 = new float[ max_SP * size_signals ];
    circular_data_c3 = new float[ max_SP * size_signals ];
    circular_parent = new int[ max_SP * size_signals ];
    circular_SNR = new float[ max_SP * size_signals ];
    fundamental = new float[ max_SP ];
    SNR = new float[ max_SP ];
    signal = new float[ size_signals ];
    visu_signal = new float[ size_signals ];
    output = new float[ size_signals ];
    fft_data = new double[ size_signals ];
    fft_buff = new double[ size_signals * max_SP ];
    variance = new double[ max_SP ];

    memset(buff_signals, 0, sizeof(float) * max_SP * size_signals );
    memset(buff_signals_c1, 0, sizeof(float) * max_SP * size_signals );
    memset(buff_signals_c2, 0, sizeof(float) * max_SP * size_signals );
    memset(buff_signals_c3, 0, sizeof(float) * max_SP * size_signals );
    memset(circular_data_c1, 0, sizeof(float) * max_SP * size_signals );
    memset(circular_data_c2, 0, sizeof(float) * max_SP * size_signals );
    memset(circular_data_c3, 0, sizeof(float) * max_SP * size_signals );
    memset(circular_SNR, 0, sizeof(float) * max_SP * size_signals );
    memset(circular_parent, 0, sizeof(int) * max_SP * size_signals );

    PBV = new double[3*max_SP];
    color = new int[ max_SP ];

}

Signal_processing::~Signal_processing() {
    delete[] buff_signals;
    delete[] buff_signals_c1;
    delete[] buff_signals_c2;
    delete[] buff_signals_c3;
    delete[] circular_data_c1;
    delete[] circular_data_c2;
    delete[] circular_data_c3;
    delete[] circular_parent;
    delete[] circular_SNR;
    delete[] fundamental;
    delete[] SNR;

    delete[] signal;
    delete[] visu_signal;
    delete[] output;
    delete[] fft_data;
    delete[] fft_buff;
    delete[] variance;
    delete[] PBV;
    delete[] color;

}

void Signal_processing::process() {
    // construct signals based on inheritance table
    if( ready ) {
        index_processed++;

        construct_signal();

        int temp_index = index_circular - 1;
        if( temp_index < 0 )
            temp_index += size_signals;

        int visu_snr = -1; //gsl_stats_max_index( fft_data, 1, nb_sp );

        for( int i=0; i<nb_sp; i++ ) {

            for( int j=0; j<size_signals; j++ ) {
                signal[ j ] = buff_signals[ max_SP * j + i ];

            }

            signal[ 0 ] = 0.f;
            signal[ size_signals-1 ] = 0.f;

            // compute fft
            for( int j=0; j<size_signals; j++ )
                fft_data[j] = double( signal[j] );

            fft(fft_data, size_signals, GSL_FFT_FORWARD);
            for( int j=0; j<size_signals; j++ ) {
                if( j > 0.8 * 2 / float( FS / float( size_signals ) ) && j < 4 * 2 / float( FS / float( size_signals ) ) )
                    fft_data[j] *= fft_data[j];
                else
                    fft_data[j] = 0;

            }

            for( int j=0; j<size_signals; j++ )
                fft_buff[ nb_sp * j + i ] = fft_data[ j ];

            // get SNR value
	    float df = 0.5*2;
            float d_width = df* float(size_signals) / float(FS);

            if( i == visu_snr )
                circular_SNR[ max_SP * temp_index + i ] = compute_SNR( fft_data, size_signals, int( round( d_width ) ), 2, 1 );
            else
                circular_SNR[ max_SP * temp_index + i ] = compute_SNR( fft_data, size_signals, int( round( d_width ) ), 2, 0 );

            SNR[ i ] += circular_SNR[ max_SP * temp_index + i ];

        }

        if( count_SNR < size_signals )
            count_SNR++;

        float weight[nb_sp] = {0.f};
        float sum_weight=0.f;
        for( int i=0; i<nb_sp; i++ ) {
            if (SNR[i] > 0)
	    {
            	SNR[i] = SNR[i] / count_SNR;

            	weight[i] = pow( 10.0, double(SNR[i])/10.0 );
            	sum_weight += weight[i];
	    }
        }

        memset( fft_data, 0, sizeof(double)*size_signals );

        for( int i=0; i<nb_sp; i++ ) {
		if (! isnan(buff_signals[max_SP * 50 + i]))
		{
            	for( int j=0; j<size_signals; j++ ) {
                    if( SNR[i] > 0 ) {
                        fft_data[j] += weight[i] * buff_signals[ max_SP * j + i ];
		}

              }
            }

        }

	// Signal final
#if VISU_SIGNAL_FINAL
	    CvPlot::clear("Signal_final");
	    CvPlot::plot("Signal_final", fft_data, size_signals);
#endif
            fft(fft_data, size_signals, GSL_FFT_FORWARD);
            for( int j=0; j<size_signals; j++ ) {
                if( j > 0.8 * 2 / float( FS / float( size_signals ) ) && j < 4 * 2 / float( FS / float( size_signals ) ) )
                    fft_data[j] *= fft_data[j];
                else
                    fft_data[j] = 0;

            }



            // get SNR value
	    float df = 0.5*2;
            float d_width = df* float(size_signals) / float(FS);

	    float SNR_final;
#if VISU_SIGNAL_FINAL 
            SNR_final = compute_SNR( fft_data, size_signals, int( round( d_width ) ), 2, 1 );
#else
            SNR_final = compute_SNR( fft_data, size_signals, int( round( d_width ) ), 2, 0 );
#endif

        HR = round( float(gsl_stats_max_index( fft_data, 1, size_signals )) * 30.f * float( FS / float( size_signals ) ) );
    }

}

const float* Signal_processing::get_SNR() {
    return SNR;

}


void Signal_processing::add_frame( int* parent, float* c_1, float* c_2, float* c_3, int nb_SP ) {
    nb_sp = nb_SP;

    for( int i=0; i<nb_sp; i++ ) {
            circular_data_c1[ max_SP * index_circular + i ] = c_1[ i ];
            circular_data_c2[ max_SP * index_circular + i ] = c_2[ i ];
            circular_data_c3[ max_SP * index_circular + i ] = c_3[ i ];

        circular_parent[ max_SP * index_circular + i ] = parent[ i ];

    }

    index_circular++;
    if( index_circular >= size_signals ) {
        index_circular = 0;
        ready = true;

    }

}

void Signal_processing::construct_signal() {
    int parent_index;
    int temp_index;
    memset( SNR, 0, sizeof(float) * max_SP );
    float c1, c2, c3;

//#pragma omp parallel for num_threads(4)
    for( int j=0; j<nb_sp; j++ ) {
         parent_index = j;

        for( int i=0; i<size_signals; i++ ) {
            temp_index = index_circular - 1 - i;
            if( temp_index < 0 )
                temp_index += size_signals;

            // construct signals
            c1 = circular_data_c1[ max_SP * temp_index + parent_index ];
            c2 = circular_data_c2[ max_SP * temp_index + parent_index ];
            c3 = circular_data_c3[ max_SP * temp_index + parent_index ];

            buff_signals_c1[ max_SP * ( size_signals - 1 - i ) + j ] = c1;
            buff_signals_c2[ max_SP * ( size_signals - 1 - i ) + j ] = c2;
            buff_signals_c3[ max_SP * ( size_signals - 1 - i ) + j ] = c3;

            parent_index = circular_parent[ max_SP * temp_index + parent_index ];

            // construct SNR_history
            if( circular_SNR[ max_SP * temp_index + parent_index ] != 0 ) {
                SNR[ j ] += circular_SNR[ max_SP * temp_index + parent_index ];
            }

        }

        // apply rPPG algo :
        double Xf[size_signals];
        double Yf[size_signals];
        double alpha;

        // C1, C2, C3
        double C1[size_signals];
        double C2[size_signals];
        double C3[size_signals];

        for(int i=0; i<size_signals; i++) {
            C1[i] = buff_signals_c1[ max_SP * i + j ];
            C2[i] = buff_signals_c2[ max_SP * i + j ];
            C3[i] = buff_signals_c3[ max_SP * i + j ];

        }

        // normalization
        double mean_C1 = gsl_stats_mean( C1, 1, size_signals );
        double sd_C1 = gsl_stats_sd( C1, 1, size_signals );

        double mean_C2 = gsl_stats_mean( C2, 1, size_signals );
        double sd_C2 = gsl_stats_sd( C2, 1, size_signals );

        double mean_C3 = gsl_stats_mean( C3, 1, size_signals );
        double sd_C3 = gsl_stats_sd( C3, 1, size_signals );

        for(int i=0; i<size_signals; i++) {
            C1[i] = ( C1[i] / mean_C1 );
            C2[i] = ( C2[i] / mean_C2 );
            C3[i] = ( C3[i] / mean_C3 );

        }

        filter( C1, size_signals, FS, output, 4, 0.4f, 4.f);
        for(int i=0; i<size_signals; i++)
            C1[i] = double(output[i]);

        filter( C2, size_signals, FS, output, 4, 0.4f, 4.f);
        for(int i=0; i<size_signals; i++)
            C2[i] = double(output[i]);

        filter( C3, size_signals, FS, output, 4, 0.4f, 4.f);
        for(int i=0; i<size_signals; i++)
            C3[i] = double(output[i]);

        //-- -- -- -- -- CHROM

        //Xs && Ys
        for(int i=0; i<size_signals; i++) {
            Xf[i] = 3*C1[i]-2*C2[i];
            Yf[i] = 1.5*C1[i]+C2[i]-1.5*C3[i];

        }

        double std_xf = gsl_stats_sd(Xf, 1, size_signals);
        double std_yf = gsl_stats_sd(Yf, 1, size_signals);

        alpha = std_xf / std_yf;

        for( int i=0; i<size_signals; i++ ) {
            buff_signals[ max_SP * i + j ] = float( Xf[i] - alpha*Yf[i] );
        }
    }
}

void Signal_processing::fft(double* data, int n, int type)
{
    gsl_fft_real_wavetable * real;
    gsl_fft_halfcomplex_wavetable * hc;
    gsl_fft_real_workspace * work;

    if(type == 1) {
        work = gsl_fft_real_workspace_alloc(n);
        real = gsl_fft_real_wavetable_alloc(n);

        gsl_fft_real_transform(data, 1, n, real, work);

        gsl_fft_real_wavetable_free(real);
        gsl_fft_real_workspace_free (work);
    }
    else if(type == 2) {
        work = gsl_fft_real_workspace_alloc(n);
        hc = gsl_fft_halfcomplex_wavetable_alloc (n);

        gsl_fft_halfcomplex_inverse (data, 1, n, hc, work);

        gsl_fft_halfcomplex_wavetable_free (hc);
        gsl_fft_real_workspace_free (work);
    }
    else
        printf("Wrong fft flag: < GSL_FFT_FORWARD || GSL_FFT_REVERSE >");
}

void Signal_processing::filter(float* signal, int length, float s, float* output, int n, float f1, float f2)
{
    f1 *= 2;
    f2 *= 2;
    float a = cos(M_PI*(f1+f2)/s)/cos(M_PI*(f2-f1)/s);
    float a2 = a*a;
    float b = tan(M_PI*(f2-f1)/s);
    float b2 = b*b;
    float r;

    int i,j;
    n = n/4;
    float *A = (float *)malloc(n*sizeof(float));
    float *d1 = (float *)malloc(n*sizeof(float));
    float *d2 = (float *)malloc(n*sizeof(float));
    float *d3 = (float *)malloc(n*sizeof(float));
    float *d4 = (float *)malloc(n*sizeof(float));
    float *w0 = (float *)calloc(n, sizeof(float));
    float *w1 = (float *)calloc(n, sizeof(float));
    float *w2 = (float *)calloc(n, sizeof(float));
    float *w3 = (float *)calloc(n, sizeof(float));
    float *w4 = (float *)calloc(n, sizeof(float));

    float C1 = 1.0;
    float C2 = 2.0;
    float C4 = 4.0;

    for(i=0; i<n; ++i) {
        r = sin(M_PI*(C2*i+C1)/(C4*n));
        s = b2 + C2*b*r + C1;

        A[i] = b2/s;
        d1[i] = C4*a*(C1+b*r)/s;
        d2[i] = C2*(b2-C2*a2-C1)/s;
        d3[i] = C4*a*(C1-b*r)/s;
        d4[i] = -(b2 - C2*b*r + C1)/s;
    }
    for(j=0; j<length; j++) {
        for(i=0; i<n; ++i) {
            w0[i] = d1[i]*w1[i] + d2[i]*w2[i]+ d3[i]*w3[i]+ d4[i]*w4[i] + signal[j];
            output[j] = A[i]*(w0[i] - C2*w2[i] + w4[i]);
            w4[i] = w3[i];
            w3[i] = w2[i];
            w2[i] = w1[i];
            w1[i] = w0[i];
        }
    }

    free(A);
    free(d1);
    free(d2);
    free(d3);
    free(d4);
    free(w0);
    free(w1);
    free(w2);
    free(w3);
    free(w4);
}

void Signal_processing::filter(double* signal, int length, float s, float* output, int n, float f1, float f2)
{
    f1 *= 2;
    f2 *= 2;
    float a = cos(M_PI*(f1+f2)/s)/cos(M_PI*(f2-f1)/s);
    float a2 = a*a;
    float b = tan(M_PI*(f2-f1)/s);
    float b2 = b*b;
    float r;

    int i,j;
    n = n/4;
    float *A = (float *)malloc(n*sizeof(float));
    float *d1 = (float *)malloc(n*sizeof(float));
    float *d2 = (float *)malloc(n*sizeof(float));
    float *d3 = (float *)malloc(n*sizeof(float));
    float *d4 = (float *)malloc(n*sizeof(float));
    float *w0 = (float *)calloc(n, sizeof(float));
    float *w1 = (float *)calloc(n, sizeof(float));
    float *w2 = (float *)calloc(n, sizeof(float));
    float *w3 = (float *)calloc(n, sizeof(float));
    float *w4 = (float *)calloc(n, sizeof(float));

    float C1 = 1.0;
    float C2 = 2.0;
    float C4 = 4.0;

    for(i=0; i<n; ++i) {
        r = sin(M_PI*(C2*i+C1)/(C4*n));
        s = b2 + C2*b*r + C1;

        A[i] = b2/s;
        d1[i] = C4*a*(C1+b*r)/s;
        d2[i] = C2*(b2-C2*a2-C1)/s;
        d3[i] = C4*a*(C1-b*r)/s;
        d4[i] = -(b2 - C2*b*r + C1)/s;
    }
    for(j=0; j<length; j++) {
        for(i=0; i<n; ++i) {
            w0[i] = d1[i]*w1[i] + d2[i]*w2[i]+ d3[i]*w3[i]+ d4[i]*w4[i] + float(signal[j]);
            output[j] = A[i]*(w0[i] - C2*w2[i] + w4[i]);
            w4[i] = w3[i];
            w3[i] = w2[i];
            w2[i] = w1[i];
            w1[i] = w0[i];
        }
    }

    free(A);
    free(d1);
    free(d2);
    free(d3);
    free(d4);
    free(w0);
    free(w1);
    free(w2);
    free(w3);
    free(w4);
}

float Signal_processing::compute_SNR(double* input, int n_input, int dirac_width, int nb_harmonic, int visu) {
    int i, j;
    int fft_plot[n_input];
    double noise_signal[n_input] = {0};
    double pure_signal[n_input] = {0};

    double int_signal = 0;
    double int_noise = 0;

    float SNR;

    //define a model
    double fft_model[n_input] = {0};
    int demi_width_dirac = dirac_width/2;

    input[0] = 0;
    int SNR_max_index = gsl_stats_max_index(input, 1, n_input);

    for(i=0; i<n_input; i++) {
        for(j=0; j<nb_harmonic; j++) {
            if(  i > ( ( (j+1) * SNR_max_index ) - demi_width_dirac ) && i < ( ( (j+1) * SNR_max_index ) + demi_width_dirac ) ) {
                fft_model[i] = 1;

                if(j>0 && SNR_max_index > 0)
                    SNR = 0;
            }
        }
    }

    for(i=0; i<n_input; i++) {
        if( fft_model[i] > 0 ) {
            int_signal += input[i];
            pure_signal[i] = input[i];
        }
        else {
            int_noise += input[i];
            noise_signal[i] = input[i];
        }
    }

    if( int_noise > 0 )
        SNR = float(10*log10(int_signal / int_noise));
    else
        SNR = 0.f;

    if(visu > 0) {
        CvPlot::clear("Visu fft");

        for(i=0; i<n_input; i++)
            fft_plot[i] = int(pure_signal[i] * 1000);

        CvPlot::plot("Visu fft", fft_plot, n_input);

        for(i=0; i<n_input; i++)
            fft_plot[i] = int(noise_signal[i] * 1000);

        CvPlot::plot("Visu fft", fft_plot, n_input);
    }

    return SNR;
}
