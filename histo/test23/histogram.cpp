/*M///////////////////////////////////////////////////////////////////////////////////////
    2 //
    3 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
    4 //
    5 //  By downloading, copying, installing or using the software you agree to this license.
    6 //  If you do not agree to this license, do not download, install,
    7 //  copy or use the software.
    8 //
    9 //
   10 //                           License Agreement
   11 //                For Open Source Computer Vision Library
   12 //
   13 // Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
   14 // Copyright (C) 2009, Willow Garage Inc., all rights reserved.
   15 // Third party copyrights are property of their respective owners.
   16 //
   17 // Redistribution and use in source and binary forms, with or without modification,
   18 // are permitted provided that the following conditions are met:
   19 //
   20 //   * Redistribution's of source code must retain the above copyright notice,
   21 //     this list of conditions and the following disclaimer.
   22 //
   23 //   * Redistribution's in binary form must reproduce the above copyright notice,
   24 //     this list of conditions and the following disclaimer in the documentation
   25 //     and/or other materials provided with the distribution.
   26 //
   27 //   * The name of the copyright holders may not be used to endorse or promote products
   28 //     derived from this software without specific prior written permission.
   29 //
   30 // This software is provided by the copyright holders and contributors "as is" and
   31 // any express or implied warranties, including, but not limited to, the implied
   32 // warranties of merchantability and fitness for a particular purpose are disclaimed.
   33 // In no event shall the Intel Corporation or contributors be liable for any direct,
   34 // indirect, incidental, special, exemplary, or consequential damages
   35 // (including, but not limited to, procurement of substitute goods or services;
   36 // loss of use, data, or profits; or business interruption) however caused
   37 // and on any theory of liability, whether in contract, strict liability,
   38 // or tort (including negligence or otherwise) arising in any way out of
   39 // the use of this software, even if advised of the possibility of such damage.
   40 //
   41 //M*/
   42 
   43 #include "precomp.hpp"
   44 
   45 using namespace cv;
   46 using namespace cv::cuda;
   47 
   48 #if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)
   49 
   50 void cv::cuda::calcHist(InputArray, OutputArray, Stream&) { throw_no_cuda(); }
   51 
   52 void cv::cuda::equalizeHist(InputArray, OutputArray, Stream&) { throw_no_cuda(); }
   53 
   54 cv::Ptr<cv::cuda::CLAHE> cv::cuda::createCLAHE(double, cv::Size) { throw_no_cuda(); return cv::Ptr<cv::cuda::CLAHE>(); }
   55 
   56 void cv::cuda::evenLevels(OutputArray, int, int, int, Stream&) { throw_no_cuda(); }
   57 
   58 void cv::cuda::histEven(InputArray, OutputArray, int, int, int, Stream&) { throw_no_cuda(); }
   59 void cv::cuda::histEven(InputArray, GpuMat*, int*, int*, int*, Stream&) { throw_no_cuda(); }
   60 
   61 void cv::cuda::histRange(InputArray, OutputArray, InputArray, Stream&) { throw_no_cuda(); }
   62 void cv::cuda::histRange(InputArray, GpuMat*, const GpuMat*, Stream&) { throw_no_cuda(); }
   63 
   64 #else /* !defined (HAVE_CUDA) */
   65 
   67 // calcHist
   68 
   69 namespace hist
   70 {
   71     void histogram256(PtrStepSzb src, int* hist, cudaStream_t stream);
   72 }
   73 
   74 void cv::cuda::calcHist(InputArray _src, OutputArray _hist, Stream& stream)
   75 {
   76     GpuMat src = _src.getGpuMat();
   77 
   78     CV_Assert( src.type() == CV_8UC1 );
   79 
   80     _hist.create(1, 256, CV_32SC1);
   81     GpuMat hist = _hist.getGpuMat();
   82 
   83     hist.setTo(Scalar::all(0), stream);
   84 
   85     hist::histogram256(src, hist.ptr<int>(), StreamAccessor::getStream(stream));
   86 }
   87 
   89 // equalizeHist
   90 
   91 namespace hist
   92 {
   93     void equalizeHist(PtrStepSzb src, PtrStepSzb dst, const int* lut, cudaStream_t stream);
   94 }
   95 
   96 void cv::cuda::equalizeHist(InputArray _src, OutputArray _dst, Stream& _stream)
   97 {
   98     GpuMat src = _src.getGpuMat();
   99 
  100     CV_Assert( src.type() == CV_8UC1 );
  101 
  102     _dst.create(src.size(), src.type());
  103     GpuMat dst = _dst.getGpuMat();
  104 
  105     int intBufSize;
  106     nppSafeCall( nppsIntegralGetBufferSize_32s(256, &intBufSize) );
  107 
  108     size_t bufSize = intBufSize + 2 * 256 * sizeof(int);
  109 
  110     BufferPool pool(_stream);
  111     GpuMat buf = pool.getBuffer(1, static_cast<int>(bufSize), CV_8UC1);
  112 
  113     GpuMat hist(1, 256, CV_32SC1, buf.data);
  114     GpuMat lut(1, 256, CV_32SC1, buf.data + 256 * sizeof(int));
  115     GpuMat intBuf(1, intBufSize, CV_8UC1, buf.data + 2 * 256 * sizeof(int));
  116 
  117     cuda::calcHist(src, hist, _stream);
  118 
  119     cudaStream_t stream = StreamAccessor::getStream(_stream);
  120     NppStreamHandler h(stream);
  121 
  122     nppSafeCall( nppsIntegral_32s(hist.ptr<Npp32s>(), lut.ptr<Npp32s>(), 256, intBuf.ptr<Npp8u>()) );
  123 
  124     hist::equalizeHist(src, dst, lut.ptr<int>(), stream);
  125 }
  126 
  128 // CLAHE
  129 
  130 namespace clahe
  131 {
  132     void calcLut(PtrStepSzb src, PtrStepb lut, int tilesX, int tilesY, int2 tileSize, int clipLimit, float lutScale, cudaStream_t stream);
  133     void transform(PtrStepSzb src, PtrStepSzb dst, PtrStepb lut, int tilesX, int tilesY, int2 tileSize, cudaStream_t stream);
  134 }
  135 
  136 namespace
  137 {
  138     class CLAHE_Impl : public cv::cuda::CLAHE
  139     {
  140     public:
  141         CLAHE_Impl(double clipLimit = 40.0, int tilesX = 8, int tilesY = 8);
  142 
  143         void apply(cv::InputArray src, cv::OutputArray dst);
  144         void apply(InputArray src, OutputArray dst, Stream& stream);
  145 
  146         void setClipLimit(double clipLimit);
  147         double getClipLimit() const;
  148 
  149         void setTilesGridSize(cv::Size tileGridSize);
  150         cv::Size getTilesGridSize() const;
  151 
  152         void collectGarbage();
  153 
  154     private:
  155         double clipLimit_;
  156         int tilesX_;
  157         int tilesY_;
  158 
  159         GpuMat srcExt_;
  160         GpuMat lut_;
  161     };
  162 
  163     CLAHE_Impl::CLAHE_Impl(double clipLimit, int tilesX, int tilesY) :
  164         clipLimit_(clipLimit), tilesX_(tilesX), tilesY_(tilesY)
  165     {
  166     }
  167 
  168     void CLAHE_Impl::apply(cv::InputArray _src, cv::OutputArray _dst)
  169     {
  170         apply(_src, _dst, Stream::Null());
  171     }
  172 
  173     void CLAHE_Impl::apply(InputArray _src, OutputArray _dst, Stream& s)
  174     {
  175         GpuMat src = _src.getGpuMat();
  176 
  177         CV_Assert( src.type() == CV_8UC1 );
  178 
  179         _dst.create( src.size(), src.type() );
  180         GpuMat dst = _dst.getGpuMat();
  181 
  182         const int histSize = 256;
  183 
  184         ensureSizeIsEnough(tilesX_ * tilesY_, histSize, CV_8UC1, lut_);
  185 
  186         cudaStream_t stream = StreamAccessor::getStream(s);
  187 
  188         cv::Size tileSize;
  189         GpuMat srcForLut;
  190 
  191         if (src.cols % tilesX_ == 0 && src.rows % tilesY_ == 0)
  192         {
  193             tileSize = cv::Size(src.cols / tilesX_, src.rows / tilesY_);
  194             srcForLut = src;
  195         }
  196         else
  197         {
  198 #ifndef HAVE_OPENCV_CUDAARITHM
  199             throw_no_cuda();
  200 #else
  201             cv::cuda::copyMakeBorder(src, srcExt_, 0, tilesY_ - (src.rows % tilesY_), 0, tilesX_ - (src.cols % tilesX_), cv::BORDER_REFLECT_101, cv::Scalar(), s);
  202 #endif
  203 
  204             tileSize = cv::Size(srcExt_.cols / tilesX_, srcExt_.rows / tilesY_);
  205             srcForLut = srcExt_;
  206         }
  207 
  208         const int tileSizeTotal = tileSize.area();
  209         const float lutScale = static_cast<float>(histSize - 1) / tileSizeTotal;
  210 
  211         int clipLimit = 0;
  212         if (clipLimit_ > 0.0)
  213         {
  214             clipLimit = static_cast<int>(clipLimit_ * tileSizeTotal / histSize);
  215             clipLimit = std::max(clipLimit, 1);
  216         }
  217 
  218         clahe::calcLut(srcForLut, lut_, tilesX_, tilesY_, make_int2(tileSize.width, tileSize.height), clipLimit, lutScale, stream);
  219 
  220         clahe::transform(src, dst, lut_, tilesX_, tilesY_, make_int2(tileSize.width, tileSize.height), stream);
  221     }
  222 
  223     void CLAHE_Impl::setClipLimit(double clipLimit)
  224     {
  225         clipLimit_ = clipLimit;
  226     }
  227 
  228     double CLAHE_Impl::getClipLimit() const
  229     {
  230         return clipLimit_;
  231     }
  232 
  233     void CLAHE_Impl::setTilesGridSize(cv::Size tileGridSize)
  234     {
  235         tilesX_ = tileGridSize.width;
  236         tilesY_ = tileGridSize.height;
  237     }
  238 
  239     cv::Size CLAHE_Impl::getTilesGridSize() const
  240     {
  241         return cv::Size(tilesX_, tilesY_);
  242     }
  243 
  244     void CLAHE_Impl::collectGarbage()
  245     {
  246         srcExt_.release();
  247         lut_.release();
  248     }
  249 }
  250 
  251 cv::Ptr<cv::cuda::CLAHE> cv::cuda::createCLAHE(double clipLimit, cv::Size tileGridSize)
  252 {
  253     return makePtr<CLAHE_Impl>(clipLimit, tileGridSize.width, tileGridSize.height);
  254 }
  255 
  257 // NPP Histogram
  258 
  259 namespace
  260 {
  261     typedef NppStatus (*get_buf_size_c1_t)(NppiSize oSizeROI, int nLevels, int* hpBufferSize);
  262     typedef NppStatus (*get_buf_size_c4_t)(NppiSize oSizeROI, int nLevels[], int* hpBufferSize);
  263 
  264     template<int SDEPTH> struct NppHistogramEvenFuncC1
  265     {
  266         typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;
  267 
  268     typedef NppStatus (*func_ptr)(const src_t* pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s * pHist,
  269             int nLevels, Npp32s nLowerLevel, Npp32s nUpperLevel, Npp8u * pBuffer);
  270     };
  271     template<int SDEPTH> struct NppHistogramEvenFuncC4
  272     {
  273         typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;
  274 
  275         typedef NppStatus (*func_ptr)(const src_t* pSrc, int nSrcStep, NppiSize oSizeROI,
  276             Npp32s * pHist[4], int nLevels[4], Npp32s nLowerLevel[4], Npp32s nUpperLevel[4], Npp8u * pBuffer);
  277     };
  278 
  279     template<int SDEPTH, typename NppHistogramEvenFuncC1<SDEPTH>::func_ptr func, get_buf_size_c1_t get_buf_size>
  280     struct NppHistogramEvenC1
  281     {
  282         typedef typename NppHistogramEvenFuncC1<SDEPTH>::src_t src_t;
  283 
  284         static void hist(const GpuMat& src, OutputArray _hist, int histSize, int lowerLevel, int upperLevel, Stream& stream)
  285         {
  286             const int levels = histSize + 1;
  287 
  288             _hist.create(1, histSize, CV_32S);
  289             GpuMat hist = _hist.getGpuMat();
  290 
  291             NppiSize sz;
  292             sz.width = src.cols;
  293             sz.height = src.rows;
  294 
  295             int buf_size;
  296             get_buf_size(sz, levels, &buf_size);
  297 
  298             BufferPool pool(stream);
  299             GpuMat buf = pool.getBuffer(1, buf_size, CV_8UC1);
  300 
  301             NppStreamHandler h(stream);
  302 
  303             nppSafeCall( func(src.ptr<src_t>(), static_cast<int>(src.step), sz, hist.ptr<Npp32s>(), levels,
  304                 lowerLevel, upperLevel, buf.ptr<Npp8u>()) );
  305 
  306             if (!stream)
  307                 cudaSafeCall( cudaDeviceSynchronize() );
  308         }
  309     };
  310     template<int SDEPTH, typename NppHistogramEvenFuncC4<SDEPTH>::func_ptr func, get_buf_size_c4_t get_buf_size>
  311     struct NppHistogramEvenC4
  312     {
  313         typedef typename NppHistogramEvenFuncC4<SDEPTH>::src_t src_t;
  314 
  315         static void hist(const GpuMat& src, GpuMat hist[4], int histSize[4], int lowerLevel[4], int upperLevel[4], Stream& stream)
  316         {
  317             int levels[] = {histSize[0] + 1, histSize[1] + 1, histSize[2] + 1, histSize[3] + 1};
  318             hist[0].create(1, histSize[0], CV_32S);
  319             hist[1].create(1, histSize[1], CV_32S);
  320             hist[2].create(1, histSize[2], CV_32S);
  321             hist[3].create(1, histSize[3], CV_32S);
  322 
  323             NppiSize sz;
  324             sz.width = src.cols;
  325             sz.height = src.rows;
  326 
  327             Npp32s* pHist[] = {hist[0].ptr<Npp32s>(), hist[1].ptr<Npp32s>(), hist[2].ptr<Npp32s>(), hist[3].ptr<Npp32s>()};
  328 
  329             int buf_size;
  330             get_buf_size(sz, levels, &buf_size);
  331 
  332             BufferPool pool(stream);
  333             GpuMat buf = pool.getBuffer(1, buf_size, CV_8UC1);
  334 
  335             NppStreamHandler h(stream);
  336 
  337             nppSafeCall( func(src.ptr<src_t>(), static_cast<int>(src.step), sz, pHist, levels, lowerLevel, upperLevel, buf.ptr<Npp8u>()) );
  338 
  339             if (!stream)
  340                 cudaSafeCall( cudaDeviceSynchronize() );
  341         }
  342     };
  343 
  344     template<int SDEPTH> struct NppHistogramRangeFuncC1
  345     {
  346         typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;
  347         typedef Npp32s level_t;
  348         enum {LEVEL_TYPE_CODE=CV_32SC1};
  349 
  350         typedef NppStatus (*func_ptr)(const src_t* pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s* pHist,
  351             const Npp32s* pLevels, int nLevels, Npp8u* pBuffer);
  352     };
  353     template<> struct NppHistogramRangeFuncC1<CV_32F>
  354     {
  355         typedef Npp32f src_t;
  356         typedef Npp32f level_t;
  357         enum {LEVEL_TYPE_CODE=CV_32FC1};
  358 
  359         typedef NppStatus (*func_ptr)(const Npp32f* pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s* pHist,
  360             const Npp32f* pLevels, int nLevels, Npp8u* pBuffer);
  361     };
  362     template<int SDEPTH> struct NppHistogramRangeFuncC4
  363     {
  364         typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;
  365         typedef Npp32s level_t;
  366         enum {LEVEL_TYPE_CODE=CV_32SC1};
  367 
  368         typedef NppStatus (*func_ptr)(const src_t* pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s* pHist[4],
  369             const Npp32s* pLevels[4], int nLevels[4], Npp8u* pBuffer);
  370     };
  371     template<> struct NppHistogramRangeFuncC4<CV_32F>
  372     {
  373         typedef Npp32f src_t;
  374         typedef Npp32f level_t;
  375         enum {LEVEL_TYPE_CODE=CV_32FC1};
  376 
  377         typedef NppStatus (*func_ptr)(const Npp32f* pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s* pHist[4],
  378             const Npp32f* pLevels[4], int nLevels[4], Npp8u* pBuffer);
  379     };
  380 
  381     template<int SDEPTH, typename NppHistogramRangeFuncC1<SDEPTH>::func_ptr func, get_buf_size_c1_t get_buf_size>
  382     struct NppHistogramRangeC1
  383     {
  384         typedef typename NppHistogramRangeFuncC1<SDEPTH>::src_t src_t;
  385         typedef typename NppHistogramRangeFuncC1<SDEPTH>::level_t level_t;
  386         enum {LEVEL_TYPE_CODE=NppHistogramRangeFuncC1<SDEPTH>::LEVEL_TYPE_CODE};
  387 
  388         static void hist(const GpuMat& src, OutputArray _hist, const GpuMat& levels, Stream& stream)
  389         {
  390             CV_Assert( levels.type() == LEVEL_TYPE_CODE && levels.rows == 1 );
  391 
  392             _hist.create(1, levels.cols - 1, CV_32S);
  393             GpuMat hist = _hist.getGpuMat();
  394 
  395             NppiSize sz;
  396             sz.width = src.cols;
  397             sz.height = src.rows;
  398 
  399             int buf_size;
  400             get_buf_size(sz, levels.cols, &buf_size);
  401 
  402             BufferPool pool(stream);
  403             GpuMat buf = pool.getBuffer(1, buf_size, CV_8UC1);
  404 
  405             NppStreamHandler h(stream);
  406 
  407             nppSafeCall( func(src.ptr<src_t>(), static_cast<int>(src.step), sz, hist.ptr<Npp32s>(), levels.ptr<level_t>(), levels.cols, buf.ptr<Npp8u>()) );
  408 
  409             if (stream == 0)
  410                 cudaSafeCall( cudaDeviceSynchronize() );
  411         }
  412     };
  413     template<int SDEPTH, typename NppHistogramRangeFuncC4<SDEPTH>::func_ptr func, get_buf_size_c4_t get_buf_size>
  414     struct NppHistogramRangeC4
  415     {
  416         typedef typename NppHistogramRangeFuncC4<SDEPTH>::src_t src_t;
  417         typedef typename NppHistogramRangeFuncC1<SDEPTH>::level_t level_t;
  418         enum {LEVEL_TYPE_CODE=NppHistogramRangeFuncC1<SDEPTH>::LEVEL_TYPE_CODE};
  419 
  420         static void hist(const GpuMat& src, GpuMat hist[4], const GpuMat levels[4], Stream& stream)
  421         {
  422             CV_Assert( levels[0].type() == LEVEL_TYPE_CODE && levels[0].rows == 1 );
  423             CV_Assert( levels[1].type() == LEVEL_TYPE_CODE && levels[1].rows == 1 );
  424             CV_Assert( levels[2].type() == LEVEL_TYPE_CODE && levels[2].rows == 1 );
  425             CV_Assert( levels[3].type() == LEVEL_TYPE_CODE && levels[3].rows == 1 );
  426 
  427             hist[0].create(1, levels[0].cols - 1, CV_32S);
  428             hist[1].create(1, levels[1].cols - 1, CV_32S);
  429             hist[2].create(1, levels[2].cols - 1, CV_32S);
  430             hist[3].create(1, levels[3].cols - 1, CV_32S);
  431 
  432             Npp32s* pHist[] = {hist[0].ptr<Npp32s>(), hist[1].ptr<Npp32s>(), hist[2].ptr<Npp32s>(), hist[3].ptr<Npp32s>()};
  433             int nLevels[] = {levels[0].cols, levels[1].cols, levels[2].cols, levels[3].cols};
  434             const level_t* pLevels[] = {levels[0].ptr<level_t>(), levels[1].ptr<level_t>(), levels[2].ptr<level_t>(), levels[3].ptr<level_t>()};
  435 
  436             NppiSize sz;
  437             sz.width = src.cols;
  438             sz.height = src.rows;
  439 
  440             int buf_size;
  441             get_buf_size(sz, nLevels, &buf_size);
  442 
  443             BufferPool pool(stream);
  444             GpuMat buf = pool.getBuffer(1, buf_size, CV_8UC1);
  445 
  446             NppStreamHandler h(stream);
  447 
  448             nppSafeCall( func(src.ptr<src_t>(), static_cast<int>(src.step), sz, pHist, pLevels, nLevels, buf.ptr<Npp8u>()) );
  449 
  450             if (stream == 0)
  451                 cudaSafeCall( cudaDeviceSynchronize() );
  452         }
  453     };
  454 }
  455 
  456 void cv::cuda::evenLevels(OutputArray _levels, int nLevels, int lowerLevel, int upperLevel, Stream& stream)
  457 {
  458     const int kind = _levels.kind();
  459 
  460     _levels.create(1, nLevels, CV_32SC1);
  461 
  462     Mat host_levels;
  463     if (kind == _InputArray::CUDA_GPU_MAT)
  464         host_levels.create(1, nLevels, CV_32SC1);
  465     else
  466         host_levels = _levels.getMat();
  467 
  468     nppSafeCall( nppiEvenLevelsHost_32s(host_levels.ptr<Npp32s>(), nLevels, lowerLevel, upperLevel) );
  469 
  470     if (kind == _InputArray::CUDA_GPU_MAT)
  471         _levels.getGpuMatRef().upload(host_levels, stream);
  472 }
  473 
  474 namespace hist
  475 {
  476     void histEven8u(PtrStepSzb src, int* hist, int binCount, int lowerLevel, int upperLevel, cudaStream_t stream);
  477 }
  478 
  479 namespace
  480 {
  481     void histEven8u(const GpuMat& src, GpuMat& hist, int histSize, int lowerLevel, int upperLevel, cudaStream_t stream)
  482     {
  483         hist.create(1, histSize, CV_32S);
  484         cudaSafeCall( cudaMemsetAsync(hist.data, 0, histSize * sizeof(int), stream) );
  485         hist::histEven8u(src, hist.ptr<int>(), histSize, lowerLevel, upperLevel, stream);
  486     }
  487 }
  488 
  489 void cv::cuda::histEven(InputArray _src, OutputArray hist, int histSize, int lowerLevel, int upperLevel, Stream& stream)
  490 {
  491     typedef void (*hist_t)(const GpuMat& src, OutputArray hist, int levels, int lowerLevel, int upperLevel, Stream& stream);
  492     static const hist_t hist_callers[] =
  493     {
  494         NppHistogramEvenC1<CV_8U , nppiHistogramEven_8u_C1R , nppiHistogramEvenGetBufferSize_8u_C1R >::hist,
  495         0,
  496         NppHistogramEvenC1<CV_16U, nppiHistogramEven_16u_C1R, nppiHistogramEvenGetBufferSize_16u_C1R>::hist,
  497         NppHistogramEvenC1<CV_16S, nppiHistogramEven_16s_C1R, nppiHistogramEvenGetBufferSize_16s_C1R>::hist
  498     };
  499 
  500     GpuMat src = _src.getGpuMat();
  501 
  502     if (src.depth() == CV_8U && deviceSupports(FEATURE_SET_COMPUTE_30))
  503     {
  504         histEven8u(src, hist.getGpuMatRef(), histSize, lowerLevel, upperLevel, StreamAccessor::getStream(stream));
  505         return;
  506     }
  507 
  508     CV_Assert( src.type() == CV_8UC1 || src.type() == CV_16UC1 || src.type() == CV_16SC1 );
  509 
  510     hist_callers[src.depth()](src, hist, histSize, lowerLevel, upperLevel, stream);
  511 }
  512 
  513 void cv::cuda::histEven(InputArray _src, GpuMat hist[4], int histSize[4], int lowerLevel[4], int upperLevel[4], Stream& stream)
  514 {
  515     typedef void (*hist_t)(const GpuMat& src, GpuMat hist[4], int levels[4], int lowerLevel[4], int upperLevel[4], Stream& stream);
  516     static const hist_t hist_callers[] =
  517     {
  518         NppHistogramEvenC4<CV_8U , nppiHistogramEven_8u_C4R , nppiHistogramEvenGetBufferSize_8u_C4R >::hist,
  519         0,
  520         NppHistogramEvenC4<CV_16U, nppiHistogramEven_16u_C4R, nppiHistogramEvenGetBufferSize_16u_C4R>::hist,
  521         NppHistogramEvenC4<CV_16S, nppiHistogramEven_16s_C4R, nppiHistogramEvenGetBufferSize_16s_C4R>::hist
  522     };
  523 
  524     GpuMat src = _src.getGpuMat();
  525 
  526     CV_Assert( src.type() == CV_8UC4 || src.type() == CV_16UC4 || src.type() == CV_16SC4 );
  527 
  528     hist_callers[src.depth()](src, hist, histSize, lowerLevel, upperLevel, stream);
  529 }
  530 
  531 void cv::cuda::histRange(InputArray _src, OutputArray hist, InputArray _levels, Stream& stream)
  532 {
  533     typedef void (*hist_t)(const GpuMat& src, OutputArray hist, const GpuMat& levels, Stream& stream);
  534     static const hist_t hist_callers[] =
  535     {
  536         NppHistogramRangeC1<CV_8U , nppiHistogramRange_8u_C1R , nppiHistogramRangeGetBufferSize_8u_C1R >::hist,
  537         0,
  538         NppHistogramRangeC1<CV_16U, nppiHistogramRange_16u_C1R, nppiHistogramRangeGetBufferSize_16u_C1R>::hist,
  539         NppHistogramRangeC1<CV_16S, nppiHistogramRange_16s_C1R, nppiHistogramRangeGetBufferSize_16s_C1R>::hist,
  540         0,
  541         NppHistogramRangeC1<CV_32F, nppiHistogramRange_32f_C1R, nppiHistogramRangeGetBufferSize_32f_C1R>::hist
  542     };
  543 
  544     GpuMat src = _src.getGpuMat();
  545     GpuMat levels = _levels.getGpuMat();
  546 
  547     CV_Assert( src.type() == CV_8UC1 || src.type() == CV_16UC1 || src.type() == CV_16SC1 || src.type() == CV_32FC1 );
  548 
  549     hist_callers[src.depth()](src, hist, levels, stream);
  550 }
  551 
  552 void cv::cuda::histRange(InputArray _src, GpuMat hist[4], const GpuMat levels[4], Stream& stream)
  553 {
  554     typedef void (*hist_t)(const GpuMat& src, GpuMat hist[4], const GpuMat levels[4], Stream& stream);
  555     static const hist_t hist_callers[] =
  556     {
  557         NppHistogramRangeC4<CV_8U , nppiHistogramRange_8u_C4R , nppiHistogramRangeGetBufferSize_8u_C4R >::hist,
  558         0,
  559         NppHistogramRangeC4<CV_16U, nppiHistogramRange_16u_C4R, nppiHistogramRangeGetBufferSize_16u_C4R>::hist,
  560         NppHistogramRangeC4<CV_16S, nppiHistogramRange_16s_C4R, nppiHistogramRangeGetBufferSize_16s_C4R>::hist,
  561         0,
  562         NppHistogramRangeC4<CV_32F, nppiHistogramRange_32f_C4R, nppiHistogramRangeGetBufferSize_32f_C4R>::hist
  563     };
  564 
  565     GpuMat src = _src.getGpuMat();
  566 
  567     CV_Assert( src.type() == CV_8UC4 || src.type() == CV_16UC4 || src.type() == CV_16SC4 || src.type() == CV_32FC4 );
  568 
  569     hist_callers[src.depth()](src, hist, levels, stream);
  570 }
  571 
  572 #endif /* !defined (HAVE_CUDA) */
