/* -- Serge Bobbia : serge.bobbia@u-bourgogne.fr -- Le2i 2018
 * This work is distributed for non commercial use only,
 * it implements the IBIS method as described in the ICPR 2018 paper.
 * The multi-threading is done with openMP, which is not an optimal solution
 * But was chosen as it was the simpliest way to do it without further justifications.
 *
 *  --------------------------- Benchmark setup : -----------------------------------------------------
 * activate or not the THREAD count from 2 to number of physical CPU core (for best performance)
 * size_roi = 9
 * MATLAB_lab = 0
 * MASK_chrono = 0
 * VISU = 0
 * VISU_all = 0
 * OUTPUT_log = 1 : if you want to get the time output, else 0
 *
 * You better want to run IBIS on a directory than a single file since the initialization of
 * every mask could be time consuming in comparison with the processing itself.
 */

#ifndef IBIS_H
#define IBIS_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <omp.h>
#include <chrono>
#include <algorithm>
#include "utils.h"

// algo debug parameters
#define THREAD_count        4       // deactivated if <= 1
#define size_roi            9       // 9 25 49 : consider adjacent seeds for pixels assignation
#define MATLAB_lab          0       // 0=>RGB2LAB : l,a,b -> 0-255. 1=>RGB2LAB : l -> 0-100; a,b -> -127:127
#define MASK_chrono         0       // 0:1 provide more informations ( complexity, process burden repartition ) : slow down the process !
#define VISU                0       // show processed pixels for each iterations
#define VISU_all            0       // for mask by mask visu of the processing : THREAD_count = 0, very slow
#define OUTPUT_log          0       // 0:1 print output log

// temporal
#define fixed_background    1

class IBIS
{

public:
    friend class MASK;

    IBIS(int maxSPNum, int compacity);
    virtual ~IBIS();

    void process( cv::Mat* img );
    void init();
    void reset();

    int getMaxSPNumber() { return maxSPNumber;}
    int getActualSPNumber() { return SPNumber; }
    int* getLabels() { return labels; }
    int* get_inheritance() { return inheritance; }
    int* get_adjacent_sp() { return adjacent_sp; }
    int* nb_adjacent_sp() { return count_adjacent; }
    float* get_Xseeds() { return Xseeds; }
    float* get_Yseeds() { return Yseeds; }
    float* get_lseeds() { return lseeds; }
    float* get_aseeds() { return aseeds; }
    float* get_bseeds() { return bseeds; }

    float get_complexity();

protected:
    void getLAB(cv::Mat *img);
    void initSeeds();
    void mask_propagate_SP();
    void mean_seeds();
    void update_adj();
    void global_mean_seeds();

    double now_ms(void);
    void enforceConnectivity();

    // temporal
    void diff_frame();
    bool creation_deletion();

private:

    class MASK {

    private:
        int* x_var;
        int* y_var;
        int* xy_var;
        int* last_px_x;
        int* last_px_y;
        int* last_px_xy;
        int* last_parent;
        int* limit_value_fill;
        int* adjacent_mask;
        int* spotted_sp;

        int count_spotted_sp;
        int count_adjacent_mask;
        int count_var;
        int count_last;
        int count_last_parent;

        int x;
        int y;
        int size;
        int mask_index;
        int angular;
        int count_SP;
        bool filled;
        bool top_mask;

        MASK* sub_mask;
        IBIS* IBIS_data;

        double Mt1, Mt2, Mt3, Mt4;

        void generate_mask();

    public:
        MASK() {

        }
        ~MASK();

        void init(int size_in, int y_in, int x_in , int mask_level, IBIS *ptr_IBIS, int* adjacent=NULL);
        void reset();
        void process( int mask_level );
        int assign_px(int y, int x, int index_xy);
        int assign_last_px(int y, int x, int index_xy);
        void assign_last();
        void fill_mask();
        bool angular_assign();
        void assign_labels( int x, int y, int index_xy, int value );
        void get_chrono( float* times );
        int get_list_sp();
        int get_adj_mask();
        int* get_spotted() { return spotted_sp; }
        int* get_mask() { return adjacent_mask; }

    protected:
        void list_sp();

    };

    // input parameters
    int size; // size of the video frame in pixels
    int width;
    int height;

    // internals buffer
    int* vertical_index;
    MASK* mask_buffer;
    int* mask_size;
    int* adjacent_sp;
    int* count_adjacent;
    int* prev_adjacent_sp;
    int* prev_count_adjacent;
    int* initial_repartition;
    int* processed;
    int* x_vec;
    int* y_vec;
    int* labels;
    float* inv;
    float* Xseeds_Sum;
    float* Yseeds_Sum;
    float* lseeds_Sum;
    float* aseeds_Sum;
    float* bseeds_Sum;
    float* countPx;
    bool* updated_px;
    int* nlabels;

    // inner parameter
    int count_mask;
    int start_xy;
    int index_mask;
    int y_limit;
    int x_limit;
    float invwt;
    int minSPSizeThreshold;
    int maxSPNumber;            // number of Sp passed by user
    int SPNumber;               // number of Sp actually created
    int SPTypicalLength;		// typical size of the width or height for a SP
    int compacity;              // compacity factor

    // seeds value
    float* Xseeds;
    float* Yseeds;
    float* lseeds;
    float* aseeds;
    float* bseeds;

    // initial seeds repartition
    float* Xseeds_init;
    float* Yseeds_init;

    // image data
    float* lvec;
    float* avec;
    float* bvec;

    // temporal data
    float* lvec_bis;
    float* avec_bis;
    float* bvec_bis;

    float* lvec_one;
    float* avec_one;
    float* bvec_one;

    float* Xseeds_prev;
    float* Yseeds_prev;
    float* lseeds_prev;
    float* aseeds_prev;
    float* bseeds_prev;

    bool img_bis;

    // difference between frame
    bool reset_state;
    int index_frame;
    int* count_diff;
    int* inheritance;
    int* elligible;

public:
    double st2, st3, st4;

};

#endif // IBIS_H
