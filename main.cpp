#include <iostream>
#include <vector>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "tensorflow/c/c_api.h"

// #define MNIST
#define MOBILENET

#ifdef MNIST
#define INPUT_WIDTH 28
#define INPUT_HEIGHT 28
#endif

#ifdef MOBILENET
#include "coco_classes.h"
#define INPUT_WIDTH 300
#define INPUT_HEIGHT 300
#endif

void NoOpDeallocator(void *data, size_t a, void *b) {}

class TF_Operation_Wrapper
{
public:
    float *output;
    std::string name;
    int index;
    int *outDim;
    int numDim;
    uint64_t outDataSize = 1;

    TF_Operation_Wrapper(std::string n, int i, int dim[], int nd) : name(n), index(i), outDim(dim), numDim(nd)
    {
        for (int i = 0; i < numDim; i++)
            outDataSize *= outDim[i];
        output = new float[outDataSize];
    }
};

class TF_SavedModel
{
public:
    TF_SavedModel(const char *dir, const char *t,
                  unsigned int numInDim, int64_t *inDim, unsigned int dataSize,
                  std::vector<TF_Operation_Wrapper *> inputOpNames, std::vector<TF_Operation_Wrapper *> outputOpNames) : m_saved_model_dir(dir),
                                                                                                                         m_tags(t),
                                                                                                                         m_numInDim(numInDim),
                                                                                                                         m_dataSize(dataSize),
                                                                                                                         m_inputOpNames(inputOpNames),
                                                                                                                         m_outputOpNames(outputOpNames)

    {
        m_numInput = m_inputOpNames.size();
        m_numOutput = m_outputOpNames.size();

        m_inDim = new int64_t[numInDim];
        for (int i = 0; i < numInDim; i++)
            m_inDim[i] = inDim[i];
    }

    ~TF_SavedModel()
    {
        // Free memory
        TF_DeleteGraph(Graph);
        TF_DeleteSession(Session, Status);
        TF_DeleteSessionOptions(SessionOpts);
        TF_DeleteStatus(Status);
    }

    int loadModel()
    {
        Graph = TF_NewGraph();
        Status = TF_NewStatus();
        SessionOpts = TF_NewSessionOptions();
        TF_Buffer *RunOpts = NULL;

        int ntags = 1;
        Session = TF_LoadSessionFromSavedModel(SessionOpts, RunOpts, m_saved_model_dir, &m_tags, ntags, Graph, NULL, Status);
        if (!TF_GetCode(Status) == TF_OK)
        {
            printf("Error:%s", TF_Message(Status));
            return -1;
        }

        //****** Get input tensor
        Input = (TF_Output *)malloc(sizeof(TF_Output) * m_numInput);
        for (int i = 0; i < m_inputOpNames.size(); i++)
        {
            TF_Output t0 = {TF_GraphOperationByName(Graph, m_inputOpNames[i]->name.c_str()), m_inputOpNames[i]->index};
            if (t0.oper == NULL)
            {
                printf("ERROR: Failed TF_GraphOperationByName: %s\n", m_inputOpNames[i]->name.c_str());
                return -1;
            }
            Input[i] = t0;
        }

        //********* Get Output tensor
        Output = (TF_Output *)malloc(sizeof(TF_Output) * m_numOutput);
        for (int i = 0; i < m_outputOpNames.size(); i++)
        {
            TF_Output t2 = {TF_GraphOperationByName(Graph, m_outputOpNames[i]->name.c_str()), m_outputOpNames[i]->index};
            if (t2.oper == NULL)
            {
                printf("ERROR: Failed TF_GraphOperationByName: %s\n", m_outputOpNames[i]->name.c_str());
                return -2;
            }
            Output[i] = t2;
        }

        return 0;
    }

    int pipeline(cv::Mat img)
    {
        //********* Allocate data for inputs & outputs
        TF_Tensor **InputValues = (TF_Tensor **)malloc(sizeof(TF_Tensor *) * m_numInput);
        TF_Tensor **OutputValues = (TF_Tensor **)malloc(sizeof(TF_Tensor *) * m_numOutput);

        TF_Tensor *in_tensor = TF_NewTensor(TF_UINT8, m_inDim, m_numInDim, img.data, m_dataSize, &NoOpDeallocator, 0);

        if (in_tensor == NULL)
        {
            printf("ERROR: Failed TF_NewTensor\n");
            return -1;
        }

        InputValues[0] = in_tensor;

        // Run the Session
        TF_SessionRun(Session, NULL, Input, InputValues, m_numInput, Output, OutputValues, m_numOutput, NULL, 0, NULL, Status);
        if (TF_GetCode(Status) != TF_OK)
        {
            printf("%s", TF_Message(Status));
            return -2;
        }

        // TODO: how to parse output dimensions
        for (int i = 0; i < m_numOutput; i++)
        {
            printf("Model Output [%d (%ld)]: ", i, m_outputOpNames[i]->outDataSize);
            float *buff = (float *)TF_TensorData(OutputValues[i]);
            int index = 0;
            for (int j = 0; j < m_outputOpNames[i]->outDataSize; j++)
            {
                printf("%.2f, ", *buff);
                m_outputOpNames[i]->output[index++] = *buff++;
            }
            printf("\n");
        }

        return 0;
    }

private:
    TF_Graph *Graph;
    TF_Session *Session;
    TF_SessionOptions *SessionOpts;
    TF_Output *Input;
    TF_Output *Output;
    TF_Status *Status;
    int64_t *m_inDim;
    uint64_t m_dataSize;
    unsigned int m_numInDim;
    unsigned int m_numInput;
    unsigned int m_numOutDim;
    unsigned int m_numOutput;
    const char *m_saved_model_dir;
    const char *m_tags;
    std::vector<TF_Operation_Wrapper *> m_inputOpNames;
    std::vector<TF_Operation_Wrapper *> m_outputOpNames;

    void test()
    {
    }
};

void softmax(float *input, size_t size)
{
    assert(0 <= size <= sizeof(input) / sizeof(double));

    int i;
    double m, sum, constant;

    m = -INFINITY;
    for (i = 0; i < size; ++i)
        if (m < input[i])
            m = input[i];

    sum = 0.0;
    for (i = 0; i < size; ++i)
        sum += exp(input[i] - m);

    constant = m + log(sum);
    for (i = 0; i < size; ++i)
        input[i] = exp(input[i] - constant);
}

void cvPreview(cv::Mat img)
{
    cv::namedWindow("Preview", cv::WINDOW_AUTOSIZE);
    cv::imshow("Preview", img);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

#ifdef MOBILENET
int main()
{
    printf("+++\n");
    const char *file = "../images/kite.jpg";
    cv::Mat img = cv::imread(file, cv::IMREAD_COLOR);

    img.convertTo(img, CV_8UC3);
    cv::resize(img, img, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), 0, 0, cv::INTER_LINEAR);

    // cvPreview(img);

    int64_t in[] = {1, INPUT_WIDTH, INPUT_HEIGHT, 3};
    unsigned int dataSize = sizeof(uint8_t) * INPUT_WIDTH * INPUT_HEIGHT * 3;
    std::vector<TF_Operation_Wrapper *> inputOpNames = {new TF_Operation_Wrapper("serving_default_input_tensor", 0, {}, 0)};
    std::vector<TF_Operation_Wrapper *> outputOpNames = {
        new TF_Operation_Wrapper("StatefulPartitionedCall", 5, new int[2]{1}, 1),
        new TF_Operation_Wrapper("StatefulPartitionedCall", 4, new int[2]{1, 100}, 2),
        new TF_Operation_Wrapper("StatefulPartitionedCall", 2, new int[2]{1, 100}, 2),
        new TF_Operation_Wrapper("StatefulPartitionedCall", 1, new int[3]{1, 100, 4}, 3)};
    TF_SavedModel *tfModel = new TF_SavedModel("ssd_mobilenet_v2_320x320/saved_model", "serve", 4, in, dataSize, inputOpNames, outputOpNames);

    int ret = 0;
    tfModel->loadModel();
    if (ret > 0)
    {
        printf("Load model failed:%d\n", ret);
        return ret;
    }

    tfModel->pipeline(img);
    if (ret > 0)
    {
        printf("Pipeline failed:%d\n", ret);
        return ret;
    }

    float threshold = 0.5f;
    int validIndex = 0;
    for (validIndex = 0; validIndex < 100; validIndex++)
        if (outputOpNames[1]->output[validIndex] < threshold)
            break;

    printf("===========Result=============\n");
    printf("Number of detections: %d\n", validIndex);
    for (int i = 0; i < validIndex; i++)
    {
        printf("[Object %d]\n", i);
        printf("Class:%s\n", COCO_CLASSES[(int)outputOpNames[2]->output[i] - 1]);
        printf("Box: %f, %f, %f, %f\n", outputOpNames[3]->output[i * 4], outputOpNames[3]->output[i * 4 + 1], outputOpNames[3]->output[i * 4 + 2], outputOpNames[3]->output[i * 4 + 3]);
        cv::rectangle(img, cv::Point(outputOpNames[3]->output[i * 4 + 1] * INPUT_WIDTH, outputOpNames[3]->output[i * 4] * INPUT_HEIGHT), cv::Point(outputOpNames[3]->output[i * 4 + 3] * INPUT_WIDTH, outputOpNames[3]->output[i * 4 + 2] * INPUT_HEIGHT), cv::Scalar(255, 0, 0), 1, cv::LineTypes::LINE_8, 0);
        printf("\n");
    }

    cvPreview(img);
}
#endif

#ifdef MNIST
int main()
{
    const char *file = "../images/three_hand.png";
    cv::Mat img = cv::imread(file, 1);

    cv::resize(img, img, cv::Size(28, 28), 0, 0, cv::INTER_LINEAR);
    img.convertTo(img, CV_32FC1);
    img = img / 255;

    // cv::namedWindow("Preview", cv::WINDOW_AUTOSIZE);
    // cv::imshow("Preview", img);
    // cv::waitKey(0);

    int64_t in[] = {1, INPUT_WIDTH, INPUT_HEIGHT};
    int64_t out[] = {1, 10};
    unsigned int dataSize = sizeof(float) * INPUT_WIDTH * INPUT_HEIGHT;

    std::vector<TF_Operation_Wrapper *> inputOpNames = {new TF_Operation_Wrapper("serving_default_flatten_input", 0, {}, 0)};
    std::vector<TF_Operation_Wrapper *> outputOpNames = {new TF_Operation_Wrapper("StatefulPartitionedCall", 0, new int[2]{1, 10}, 2)};
    TF_SavedModel *tfModel = new TF_SavedModel("mnist_saved_model", "serve", 3, in, dataSize, inputOpNames, outputOpNames);

    int ret = 0;
    tfModel->loadModel();
    if (ret > 0)
    {
        printf("Load model failed:%d\n", ret);
        return ret;
    }

    tfModel->pipeline(img);
    if (ret > 0)
    {
        printf("Pipeline failed:%d\n", ret);
        return ret;
    }
    softmax(outputOpNames[0]->output, 10);

    int max = 0;
    for (int i = 0; i < 10; i++)
        if (outputOpNames[0]->output[i] > outputOpNames[0]->output[max])
            max = i;
    printf("result: %d\n", max);

    return 0;
}
#endif