
#include "Halide.h"


namespace {


using namespace Halide;
using namespace Halide::ConciseCasts;

Func bilateral_pixel(Func a, Func b) {
    Var x ("x"), y ("y"), c ("c");
    Func out{ "bilateral_pixel_out"};
    out(x, y) =  b(x/2, y/2) + i8_sat( a(x, y,0) - 128 )/4;
    return out;
}



class BilateralUpsampleStep : public Halide::Generator<BilateralUpsampleStep> {
public:

    // rgb must be 4x the size of mask
    Input<Func >  rgb{"rgb", UInt(8),  3};
    Input<Func >  mask{"mask",UInt(8),  2};
    Input<Func > k{"k", 2};
    Output<Func> output{"output", UInt(8), 2};


    Var x{"x"};
    Var xi{"xi"}, xo{"xo"};
    Var  y{"y"};
    Var dx{"dx"};
    Var dy{"dy"};
    Var c{ "c" };

    Func normalize_bilateral{ "normalize_bilateral"};
    Func sum_bilateral { "sum_bilateral"};


    // Defines outputs using inputs
    void generate() {

    Func weights("bilateral_weights");
    Func total_weights("bilateral_total_weights");
    Func bilateral("bilateral");



    RDom r(-4, 9, -4, 9);

    Expr dist_r = Halide::ConciseCasts::f32(Halide::ConciseCasts::i32(rgb(x, y, 0)) - Halide::ConciseCasts::i32(rgb((x + dx), (y + dy), 0)));
    Expr dist_g = Halide::ConciseCasts::f32(Halide::ConciseCasts::i32(rgb(x, y, 1)) - Halide::ConciseCasts::i32(rgb((x + dx), (y + dy), 1)));
    Expr dist_b = Halide::ConciseCasts::f32(Halide::ConciseCasts::i32(rgb(x, y, 2)) - Halide::ConciseCasts::i32(rgb((x + dx), (y + dy), 2)));

     Expr dist =  (dist_r * dist_r +  dist_g * dist_g + dist_b*dist_b) / 3.f;
     // Func gray_scale;
     //gray_scale(x,y) =  u8_sat(rgb(x,y,0)/3 + rgb(x,y,1)/3 + rgb(x,y,2) /3)  ;
    // Expr dist = Halide::ConciseCasts::f32(gray_scale(x,y) - gray_scale((x + dx), (y + dy)))   ;

    //float sig2 = 100.f;                 // 2 * sigma ^ 2
    float sigma_r = 10;

    // score represents the weight contribution due to intensity difference
    weights(dx, dy, x, y ) = exp(-(  (dist ) / (2 * sigma_r * sigma_r ))) * k(dx, dy) ;

    normalize_bilateral(x, y) = 0.0f ;
    sum_bilateral(x, y) = 0.0f ;

    normalize_bilateral(x, y) +=  (  weights(r.x, r.y, x, y )) ;
      sum_bilateral(x, y) +=      (  weights(r.x, r.y, x, y ) * mask((x + r.x)/( 2 ), (y + r.y)/( 2 ) )  ) ;

    // output normalizes weights to total weights

    //bilateral(x, y ) = sum(mask((x + r.x)/( 2 ), (y + r.y)/( 2 ) )  * weights(r.x, r.y, x, y )) / total_weights(x, y );

    bilateral(x, y ) = sum_bilateral(x, y ) / normalize_bilateral(x, y ) ;
    output(x, y ) =  Halide::ConciseCasts::u8_sat(bilateral(x, y ) ) ;


    }

    // Defines how inputs are scheduled.
    void schedule() {

        normalize_bilateral.compute_root();
        normalize_bilateral.update(0).split(x, xo, xi, 16).vectorize(xi);

        sum_bilateral.compute_root();
        sum_bilateral.update(0).split(x, xo, xi, 16).vectorize(xi);

        output.compute_root().parallel(y).vectorize(x, 16);

    }
};


class BilateralUpsample : public Generator<BilateralUpsample>{

  private:
        Var x{"x"}, y{"y"}, c{"c"};
        std::unique_ptr<BilateralUpsampleStep> upsample_1;
        std::unique_ptr<BilateralUpsampleStep> upsample_2;
        std::unique_ptr<BilateralUpsampleStep> upsample_3;
  public:
        Input<Buffer<uint8_t>>  rgb{"rgb",  3};
        Input<Buffer<uint8_t>>  mask{"mask", 2};
        Input<Buffer<float>> k{"k", 2};
        Output<Func> output{"output", UInt(8), 2};


     void generate() {

            Func rgb_mirror = BoundaryConditions::mirror_image(rgb   );
            Func mask_mirror = BoundaryConditions::mirror_image(mask  );

            Func down_sample_2("down_sample_2");
            Func down_sample_4("down_sample_4");
            Func bilateral_output("bilateral_output");

            upsample_1 = create<BilateralUpsampleStep>();
            upsample_2 = create<BilateralUpsampleStep>();
            upsample_3 = create<BilateralUpsampleStep>();

            down_sample_2(x,y,c) = (rgb_mirror(2*x, 2*y, c)/4 + rgb_mirror(2*x+1, 2*y, c)/4 + rgb_mirror(2*x, 2*y+1, c)/4 + rgb_mirror(2*x+1, 2*y+1, c)/4);

            down_sample_4(x,y,c) = (down_sample_2(2*x, 2*y, c)/4 + down_sample_2(2*x+1, 2*y, c)/4 + down_sample_2(2*x, 2*y+1, c)/4 + down_sample_2(2*x+1, 2*y+1, c)/4);

            upsample_3->apply(down_sample_4, mask_mirror ,  k );
            upsample_2->apply(down_sample_2,  upsample_3->output  , k  );
            upsample_1->apply(rgb_mirror, upsample_2->output , k  );
            bilateral_output(x, y ) =   upsample_1->output(x,y);

            Expr px = 15*(bilateral_output(x, y ) / 255.0f - 0.5f);
            Expr sig = 255.0f / (1.0f + exp(-px));
            output(x, y ) = u8_sat(sig);

         }

  void schedule() {

        output.compute_root();
  }
};

}  // namespace