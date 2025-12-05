#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <ctype.h>

#define MAX_EPOCHS 1000
#define MAX_SAMPLES 1000
#define MAX_FEATURES 10
#define MAX_MODELS 10

// Enumeración de algoritmos disponibles
typedef enum {
    ALGO_PERCEPTRON_SIMPLE,
    ALGO_KNN,                 
    ALGO_MLP_BASICO           
} AlgorithmType;

// Enumeración de funciones de activación
typedef enum {
    ACTIVATION_SIGMOIDE,
    ACTIVATION_TANH, 
    ACTIVATION_RELU,
    ACTIVATION_STEP
} ActivationFunction;

// Estructura para el perceptrón simple
typedef struct {
    double weights[MAX_FEATURES];
    double bias;
    int num_features;
    ActivationFunction activation;
} PerceptronSimple;

// Estructura básica para MLP
typedef struct {
    double weights_layer1[MAX_FEATURES][5];
    double weights_layer2[5];                
    double bias_layer1[5];
    double bias_layer2;
    int num_features;
    int num_hidden;
} MLPBasico;

// Estructura para K-NN
typedef struct {
    double samples[MAX_SAMPLES][MAX_FEATURES];
    int targets[MAX_SAMPLES];
    int num_samples;
    int num_features;
    int k;
} KNNModel;

// Estructura principal del modelo
typedef struct {
    AlgorithmType algorithm;
    char name[50];
    double learning_rate;
    double error_history[MAX_EPOCHS];
    int epochs_trained;
    int num_features;
    
    union {
        PerceptronSimple perceptron;
        MLPBasico mlp_basico;
        KNNModel knn_model;
    } model;
    
    time_t created_at;
    time_t last_trained;
} Model;

// Estructura para datos de entrenamiento
typedef struct {
    double features[MAX_FEATURES];
    int target;
    char label[50];
} TrainingData;

// Prototipos de funciones
void initialize_model(Model *model, AlgorithmType algo, int num_features);
int save_model(const Model *model, const char *filename);
int load_model(Model *model, const char *filename);
void print_model_info(const Model *model);
void train_perceptron_simple(Model *model, TrainingData data[], int num_samples);
void train_mlp_basico(Model *model, TrainingData data[], int num_samples);
void train_knn(Model *model, TrainingData data[], int num_samples);
int predict(const Model *model, const double features[]);
double predict_proba(const Model *model, const double features[]);
double sigmoid(double x);
double tanh_activation(double x);
double relu(double x);
double step_function(double x);
double activation_function(ActivationFunction func, double x);
double activation_derivative(ActivationFunction func, double x);

// Nuevas funciones de visualización mejoradas
void show_training_dashboard(Model *model, TrainingData data[], int num_samples, 
                           int current_epoch, double current_error);
void plot_error_evolution(const Model *model, int current_epoch);
void show_weights_and_bias(const Model *model);
void show_classification_plane(const Model *model, TrainingData data[], int num_samples);
void show_confidence_map_enhanced(const Model *model, TrainingData data[], int num_samples);
void print_confusion_matrix_enhanced(const Model *model, TrainingData data[], int num_samples);

void show_data_analysis(TrainingData data[], int num_samples, int num_features);
int load_training_data(const char *filename, TrainingData data[], int max_samples);
void clear_screen();
void print_help();
void show_comparison_metrics(Model *models[], int num_models, TrainingData data[], int num_samples);
double calculate_accuracy(const Model *model, TrainingData data[], int num_samples);
void print_confusion_matrix(const Model *model, TrainingData data[], int num_samples);

// Funciones para problemas predefinidos
void generate_and_data(TrainingData data[], int *num_samples);
void generate_or_data(TrainingData data[], int *num_samples);
void generate_xor_data(TrainingData data[], int *num_samples);
void show_problem_menu();

// Funciones para MLP básico
double forward_pass_mlp(const MLPBasico *mlp, const double features[]);
void backpropagate_mlp(MLPBasico *mlp, const double features[], int target, double learning_rate);

// Nueva función: Modo automático de entrenamiento y uso
void automatic_mode(const char *data_filename);

// ============================================================================
// NUEVAS FUNCIONES DE VISUALIZACIÓN MEJORADAS
// ============================================================================

void show_training_dashboard(Model *model, TrainingData data[], int num_samples, 
                           int current_epoch, double current_error) {
    clear_screen();
    
    // Encabezado principal
    printf("┌─────────────────────────────────────────────────────────┐\n");
    printf("│ 🧠 %-47s │\n", model->name);
    printf("├─────────────────────────────────────────────────────────┤\n");
    
    // Información de entrenamiento
    int progress = (int)((double)(current_epoch + 1) / MAX_EPOCHS * 30);
    printf("│ Época: %4d/%-4d | Error: %.4f | LR: %.2f              │\n", 
           current_epoch + 1, MAX_EPOCHS, current_error, model->learning_rate);
    printf("│ [");
    for(int i = 0; i < 30; i++) {
        if(i < progress) printf("█");
        else printf(" ");
    }
    printf("] %3d%% │\n", (int)((double)(current_epoch + 1) / MAX_EPOCHS * 100));
    
    printf("├─────────────────────────────────────────────────────────┤\n");
    
    // Evolución del error
    printf("│ 📈 EVOLUCIÓN DEL ERROR                                 │\n");
    plot_error_evolution(model, current_epoch);
    
    printf("├─────────────────────────────────────────────────────────┤\n");
    
    // Pesos y bias (solo para perceptrón)
    if(model->algorithm == ALGO_PERCEPTRON_SIMPLE) {
        show_weights_and_bias(model);
        printf("├─────────────────────────────────────────────────────────┤\n");
    }
    
    // Plano de clasificación
    printf("│ 🗺️  PLANO DE CLASIFICACIÓN                            │\n");
    show_classification_plane(model, data, num_samples);
    
    printf("├─────────────────────────────────────────────────────────┤\n");
    
    // Matriz de confusión
    printf("│ 📊 MATRIZ DE CONFUSIÓN                                │\n");
    print_confusion_matrix_enhanced(model, data, num_samples);
    
    printf("└─────────────────────────────────────────────────────────┘\n");
    
    printf("\nPresiona Ctrl+C para detener...\n");
}

void plot_error_evolution(const Model *model, int current_epoch) {
    if(current_epoch < 1) return;
    
    int display_points = 50;
    int start_epoch = (current_epoch > display_points) ? current_epoch - display_points : 0;
    int points_to_show = current_epoch - start_epoch + 1;
    
    // Encontrar el rango de errores para escalar
    double min_error = 1e10, max_error = -1e10;
    for(int i = start_epoch; i <= current_epoch; i++) {
        if(model->error_history[i] < min_error) min_error = model->error_history[i];
        if(model->error_history[i] > max_error) max_error = model->error_history[i];
    }
    
    // Asegurar que haya un rango visible
    if(max_error - min_error < 0.001) {
        min_error -= 0.005;
        max_error += 0.005;
    }
    
    int graph_height = 6;
    
    // Crear gráfico
    for(int line = graph_height; line >= 0; line--) {
        double level = min_error + (max_error - min_error) * line / graph_height;
        
        if(line == graph_height) {
            printf("│ %6.3f ┤", max_error);
        } else if(line == 0) {
            printf("│ %6.3f ┤", min_error);
        } else {
            printf("│       ┤");
        }
        
        for(int i = 0; i < points_to_show; i++) {
            int epoch = start_epoch + i;
            double error = model->error_history[epoch];
            
            double normalized = (error - min_error) / (max_error - min_error);
            int error_line = (int)(normalized * graph_height);
            
            if(error_line == line) {
                printf("█");
            } else if(error_line > line && error_line <= line + 2) {
                printf("▒");
            } else {
                printf(" ");
            }
        }
        printf(" │\n");
    }
    
    // Eje X
    printf("│        ");
    for(int i = 0; i < points_to_show; i++) {
        if(i == 0 || i == points_to_show - 1) {
            printf("┼");
        } else {
            printf("─");
        }
    }
    printf("───────── │\n");
    
    // Etiquetas del eje X
    printf("│         %-4d", start_epoch);
    for(int i = 0; i < points_to_show - 8; i++) printf(" ");
    printf("%4d       │\n", current_epoch);
}

void show_weights_and_bias(const Model *model) {
    printf("│ ⚖️  PESOS Y BIAS (Normalizados)                        │\n");
    
    if(model->algorithm != ALGO_PERCEPTRON_SIMPLE) return;
    
    // Encontrar el valor máximo para normalizar
    double max_abs = 0;
    for(int i = 0; i < model->num_features; i++) {
        double abs_val = fabs(model->model.perceptron.weights[i]);
        if(abs_val > max_abs) max_abs = abs_val;
    }
    double bias_abs = fabs(model->model.perceptron.bias);
    if(bias_abs > max_abs) max_abs = bias_abs;
    
    if(max_abs < 1e-10) max_abs = 1.0; // Evitar división por cero
    
    int bar_width = 20;
    
    for(int i = 0; i < model->num_features; i++) {
        double weight = model->model.perceptron.weights[i];
        int filled = (int)(fabs(weight) / max_abs * bar_width);
        
        printf("│ w%d: ", i + 1);
        for(int j = 0; j < bar_width; j++) {
            if(j < filled) printf("█");
            else printf(".");
        }
        printf(" %6.3f │\n", weight);
    }
    
    // Mostrar bias
    double bias = model->model.perceptron.bias;
    int bias_filled = (int)(fabs(bias) / max_abs * bar_width);
    
    printf("│ Bias:");
    for(int j = 0; j < bar_width; j++) {
        if(j < bias_filled) printf("█");
        else printf(".");
    }
    printf(" %6.3f │\n", bias);
}

void show_classification_plane(const Model *model, TrainingData data[], int num_samples) {
    int grid_size = 9; // 0-8 para que quepa en el ancho
    
    printf("│ y     ");
    for(int x = 0; x < grid_size; x++) printf("%d ", x);
    printf("  x │\n");
    
    for(int y = grid_size - 1; y >= 0; y--) {
        printf("│ %1d   ", y);
        
        for(int x = 0; x < grid_size; x++) {
            // Convertir coordenadas de grid a características normalizadas
            double fx = (double)x / (grid_size - 1);
            double fy = (double)y / (grid_size - 1);
            double features[2] = {fx, fy};
            
            // Predecir
            int prediction = predict(model, features);
            double confidence = predict_proba(model, features);
            
            // Buscar si hay un punto de datos en esta posición
            char symbol = ' ';
            int has_data_point = 0;
            
            for(int i = 0; i < num_samples; i++) {
                double dx = fabs(data[i].features[0] - fx);
                double dy = fabs(data[i].features[1] - fy);
                if(dx < 0.1 && dy < 0.1) {
                    has_data_point = 1;
                    symbol = (data[i].target == 0) ? 'O' : 'X';
                    break;
                }
            }
            
            if(!has_data_point) {
                // Mostrar región de clasificación
                if(confidence > 0.7) symbol = (prediction == 1) ? '#' : '.';
                else if(confidence > 0.6) symbol = (prediction == 1) ? '*' : ',';
                else symbol = ' ';
            }
            
            printf("%c ", symbol);
        }
        printf(" │\n");
    }
}

void show_confidence_map_enhanced(const Model *model, TrainingData data[], int num_samples) {
    printf("┌─────────────────────────────────────────────────────────┐\n");
    printf("│ 🎨 MAPA DE CONFIANZA - VISUALIZACIÓN 2D               │\n");
    printf("├─────────────────────────────────────────────────────────┤\n");
    
    int grid_size = 12;
    
    printf("│     ");
    for(int x = 0; x < grid_size; x++) printf("%d ", x % 10);
    printf("    │\n");
    
    for(int y = grid_size - 1; y >= 0; y--) {
        printf("│ %2d  ", y);
        
        for(int x = 0; x < grid_size; x++) {
            double fx = (double)x / (grid_size - 1);
            double fy = (double)y / (grid_size - 1);
            double features[2] = {fx, fy};
            
            double confidence = predict_proba(model, features);
            char symbol;
            
            if(confidence < 0.3) symbol = ' ';
            else if(confidence < 0.4) symbol = '.';
            else if(confidence < 0.5) symbol = ':';
            else if(confidence < 0.6) symbol = '-';
            else if(confidence < 0.7) symbol = '=';
            else if(confidence < 0.8) symbol = '+';
            else if(confidence < 0.9) symbol = '*';
            else symbol = '#';
            
            printf("%c ", symbol);
        }
        printf(" │\n");
    }
    
    printf("├─────────────────────────────────────────────────────────┤\n");
    printf("│ 🔍 LEYENDA:                                            │\n");
    printf("│    ' '=0%%  '.'=30%%  ':'=40%%  '-'=50%%                 │\n");
    printf("│    '='=60%%  '+'=70%%  '*'=80%%  '#'=90%%+              │\n");
    printf("└─────────────────────────────────────────────────────────┘\n");
}

void print_confusion_matrix_enhanced(const Model *model, TrainingData data[], int num_samples) {
    int true_positive = 0, true_negative = 0, false_positive = 0, false_negative = 0;
    
    for(int i = 0; i < num_samples; i++) {
        int prediction = predict(model, data[i].features);
        int actual = data[i].target;
        
        if(actual == 1 && prediction == 1) true_positive++;
        else if(actual == 0 && prediction == 0) true_negative++;
        else if(actual == 0 && prediction == 1) false_positive++;
        else if(actual == 1 && prediction == 0) false_negative++;
    }
    
    double accuracy = (double)(true_positive + true_negative) / num_samples * 100;
    double precision = (true_positive + false_positive) > 0 ? 
                      (double)true_positive / (true_positive + false_positive) * 100 : 0;
    double recall = (true_positive + false_negative) > 0 ? 
                   (double)true_positive / (true_positive + false_negative) * 100 : 0;
    
    printf("│         Pred 0    Pred 1    │\n");
    printf("│ Real 0:   %2d        %2d      │\n", true_negative, false_positive);
    printf("│ Real 1:   %2d        %2d      │\n", false_negative, true_positive);
    printf("│ Precisión: %.1f%%              │\n", accuracy);
    printf("│ Recall:    %.1f%%              │\n", recall);
}

// ============================================================================
// FUNCIONES ORIGINALES MODIFICADAS
// ============================================================================

void show_data_analysis(TrainingData data[], int num_samples, int num_features) {
    printf("┌─────────────────────────────────────────────────────────┐\n");
    printf("│ 📊 ANÁLISIS DE DATOS CARGADOS                         │\n");
    printf("├─────────────────────────────────────────────────────────┤\n");
    
    int class_0 = 0, class_1 = 0;
    for(int i = 0; i < num_samples; i++) {
        if(data[i].target == 0) class_0++;
        else class_1++;
    }
    
    printf("│ Muestras totales: %-34d │\n", num_samples);
    printf("│ Clase 0: %-3d (%-5.1f%%)                              │\n", 
           class_0, (double)class_0/num_samples*100);
    printf("│ Clase 1: %-3d (%-5.1f%%)                              │\n", 
           class_1, (double)class_1/num_samples*100);
    
    printf("├─────────────────────────────────────────────────────────┤\n");
    printf("│ Primeras muestras:                                      │\n");
    for(int i = 0; i < (num_samples < 3 ? num_samples : 3); i++) {
        printf("│   [%.1f, %.1f] -> %-2d                                │\n", 
               data[i].features[0], data[i].features[1], data[i].target);
    }
    printf("└─────────────────────────────────────────────────────────┘\n");
}

double calculate_accuracy(const Model *model, TrainingData data[], int num_samples) {
    int correct = 0;
    for(int i = 0; i < num_samples; i++) {
        int prediction = predict(model, data[i].features);
        if(prediction == data[i].target) {
            correct++;
        }
    }
    return (double)correct / num_samples;
}

void print_confusion_matrix(const Model *model, TrainingData data[], int num_samples) {
    print_confusion_matrix_enhanced(model, data, num_samples);
}

void show_comparison_metrics(Model *models[], int num_models, TrainingData data[], int num_samples) {
    printf("┌─────────────────────────────────────────────────────────┐\n");
    printf("│ 📊 COMPARACIÓN DE MODELOS                             │\n");
    printf("├───────────────┬────────────┬────────────┬───────────────┤\n");
    printf("│ Modelo        │ Precisión  │ Épocas     │ Error Final   │\n");
    printf("├───────────────┼────────────┼────────────┼───────────────┤\n");
    
    for(int i = 0; i < num_models; i++) {
        double accuracy = calculate_accuracy(models[i], data, num_samples);
        double final_error = models[i]->epochs_trained > 0 ? 
                           models[i]->error_history[models[i]->epochs_trained - 1] : 0;
        
        printf("│ %-13s │ %8.1f%%  │ %10d │ %13.4f │\n",
               models[i]->name, accuracy * 100, 
               models[i]->epochs_trained, final_error);
    }
    printf("└───────────────┴────────────┴────────────┴───────────────┘\n");
}

// Implementación de funciones para problemas predefinidos
void generate_and_data(TrainingData data[], int *num_samples) {
    data[0].features[0] = 0.0; data[0].features[1] = 0.0; data[0].target = 0;
    data[1].features[0] = 0.0; data[1].features[1] = 1.0; data[1].target = 0;
    data[2].features[0] = 1.0; data[2].features[1] = 0.0; data[2].target = 0;
    data[3].features[0] = 1.0; data[3].features[1] = 1.0; data[3].target = 1;
    *num_samples = 4;
}

void generate_or_data(TrainingData data[], int *num_samples) {
    data[0].features[0] = 0.0; data[0].features[1] = 0.0; data[0].target = 0;
    data[1].features[0] = 0.0; data[1].features[1] = 1.0; data[1].target = 1;
    data[2].features[0] = 1.0; data[2].features[1] = 0.0; data[2].target = 1;
    data[3].features[0] = 1.0; data[3].features[1] = 1.0; data[3].target = 1;
    *num_samples = 4;
}

void generate_xor_data(TrainingData data[], int *num_samples) {
    data[0].features[0] = 0.0; data[0].features[1] = 0.0; data[0].target = 0;
    data[1].features[0] = 0.0; data[1].features[1] = 1.0; data[1].target = 1;
    data[2].features[0] = 1.0; data[2].features[1] = 0.0; data[2].target = 1;
    data[3].features[0] = 1.0; data[3].features[1] = 1.0; data[3].target = 0;
    *num_samples = 4;
}

void show_problem_menu() {
    printf("┌─────────────────────────────────────────────────────────┐\n");
    printf("│ 🎯 PROBLEMAS PREDEFINIDOS                             │\n");
    printf("├─────────────────────────────────────────────────────────┤\n");
    printf("│ 1. AND (Y Lógico) - Clasificación linealmente separable │\n");
    printf("│ 2. OR (O Lógico) - Clasificación linealmente separable  │\n");
    printf("│ 3. XOR (O Exclusivo) - Clasificación no lineal          │\n");
    printf("├─────────────────────────────────────────────────────────┤\n");
    printf("│ Ejemplos de uso:                                        │\n");
    printf("│   ./programa --entrenar --problema AND --modelo and.bin │\n");
    printf("│   ./programa --entrenar --problema OR --algoritmo perceptron │\n");
    printf("│   ./programa --entrenar --problema XOR --algoritmo mlp  │\n");
    printf("└─────────────────────────────────────────────────────────┘\n");
}

// Implementación de funciones de gestión de modelos
void initialize_model(Model *model, AlgorithmType algo, int num_features) {
    model->algorithm = algo;
    model->num_features = num_features;
    model->learning_rate = 0.1;
    model->epochs_trained = 0;
    model->created_at = time(NULL);
    model->last_trained = time(NULL);
    
    switch(algo) {
        case ALGO_PERCEPTRON_SIMPLE:
            strcpy(model->name, "Perceptrón Simple");
            for(int i = 0; i < num_features; i++) {
                model->model.perceptron.weights[i] = ((double)rand() / RAND_MAX) * 2 - 1;
            }
            model->model.perceptron.bias = ((double)rand() / RAND_MAX) * 2 - 1;
            model->model.perceptron.num_features = num_features;
            model->model.perceptron.activation = ACTIVATION_SIGMOIDE;
            break;
            
        case ALGO_MLP_BASICO:
            strcpy(model->name, "MLP Básico (2 capas)");
            model->model.mlp_basico.num_features = num_features;
            model->model.mlp_basico.num_hidden = 5;
            
            // Inicializar pesos capa entrada->oculta
            for(int i = 0; i < num_features; i++) {
                for(int j = 0; j < 5; j++) {
                    model->model.mlp_basico.weights_layer1[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
                }
            }
            
            // Inicializar pesos capa oculta->salida
            for(int j = 0; j < 5; j++) {
                model->model.mlp_basico.weights_layer2[j] = ((double)rand() / RAND_MAX) * 2 - 1;
            }
            
            // Inicializar biases
            for(int j = 0; j < 5; j++) {
                model->model.mlp_basico.bias_layer1[j] = ((double)rand() / RAND_MAX) * 2 - 1;
            }
            model->model.mlp_basico.bias_layer2 = ((double)rand() / RAND_MAX) * 2 - 1;
            break;
            
        case ALGO_KNN:
            strcpy(model->name, "K-Vecinos Más Cercanos");
            model->model.knn_model.k = 3;
            model->model.knn_model.num_samples = 0;
            model->model.knn_model.num_features = num_features;
            break;
    }
    
    for(int i = 0; i < MAX_EPOCHS; i++) {
        model->error_history[i] = 0.0;
    }
}

int save_model(const Model *model, const char *filename) {
    FILE *file = fopen(filename, "wb");
    if(!file) return 0;
    
    int result = fwrite(model, sizeof(Model), 1, file);
    fclose(file);
    
    return result == 1;
}

int load_model(Model *model, const char *filename) {
    FILE *file = fopen(filename, "rb");
    if(!file) return 0;
    
    int result = fread(model, sizeof(Model), 1, file);
    fclose(file);
    
    return result == 1;
}

void print_model_info(const Model *model) {
    printf("┌─────────────────────────────────────────────────────────┐\n");
    printf("│ 🤖 INFORMACIÓN DEL MODELO                             │\n");
    printf("├─────────────────────────────────────────────────────────┤\n");
    printf("│ Nombre: %-45s │\n", model->name);
    printf("│ Algoritmo: ");
    switch(model->algorithm) {
        case ALGO_PERCEPTRON_SIMPLE: printf("Perceptrón Simple%-29s │\n", ""); break;
        case ALGO_MLP_BASICO: printf("MLP Básico (2 capas)%-26s │\n", ""); break;
        case ALGO_KNN: printf("K-Vecinos Más Cercanos%-24s │\n", ""); break;
    }
    printf("│ Características: %-36d │\n", model->num_features);
    printf("│ Épocas entrenadas: %-34d │\n", model->epochs_trained);
    printf("│ Tasa aprendizaje: %-35.3f │\n", model->learning_rate);
    
    char time_buf[26];
    struct tm* tm_info = localtime(&model->created_at);
    strftime(time_buf, 26, "%Y-%m-%d %H:%M:%S", tm_info);
    printf("│ Creado: %-42s │\n", time_buf);
    
    tm_info = localtime(&model->last_trained);
    strftime(time_buf, 26, "%Y-%m-%d %H:%M:%S", tm_info);
    printf("│ Último entrenamiento: %-31s │\n", time_buf);
    
    if(model->epochs_trained > 0) {
        printf("│ Error final: %-38.4f │\n", model->error_history[model->epochs_trained - 1]);
    }
    printf("└─────────────────────────────────────────────────────────┘\n");
}

// Implementación de algoritmos de ML
void train_perceptron_simple(Model *model, TrainingData data[], int num_samples) {
    printf("┌─────────────────────────────────────────────────────────┐\n");
    printf("│ 🧠 ENTRENANDO PERCEPTRÓN SIMPLE                       │\n");
    printf("├─────────────────────────────────────────────────────────┤\n");
    printf("│ ℹ️  Funciona bien para problemas linealmente separables │\n");
    printf("│    (AND, OR)                                           │\n");
    printf("└─────────────────────────────────────────────────────────┘\n\n");
    
    int convergence = 0;
    for(int epoch = 0; epoch < MAX_EPOCHS && !convergence; epoch++) {
        double total_error = 0.0;
        convergence = 1;
        
        for(int i = 0; i < num_samples; i++) {
            double z = model->model.perceptron.bias;
            for(int j = 0; j < model->num_features; j++) {
                z += model->model.perceptron.weights[j] * data[i].features[j];
            }
            
            double prediction = activation_function(model->model.perceptron.activation, z);
            double error = data[i].target - prediction;
            
            if(fabs(error) > 0.1) {
                convergence = 0;
                
                double delta = model->learning_rate * error * 
                              activation_derivative(model->model.perceptron.activation, z);
                
                model->model.perceptron.bias += delta;
                for(int j = 0; j < model->num_features; j++) {
                    model->model.perceptron.weights[j] += delta * data[i].features[j];
                }
            }
            
            total_error += error * error;
        }
        
        total_error /= num_samples;
        model->error_history[epoch] = total_error;
        model->epochs_trained = epoch + 1;
        
        if(epoch % 10 == 0 || epoch == MAX_EPOCHS - 1 || convergence) {
            show_training_dashboard(model, data, num_samples, epoch, total_error);
            usleep(200000);
        }
        
        if(total_error < 0.01) {
            convergence = 1;
        }
    }
    
    model->last_trained = time(NULL);
    
    printf("┌─────────────────────────────────────────────────────────┐\n");
    printf("│ ⚠️  ENTRENAMIENTO COMPLETADO                           │\n");
    printf("├─────────────────────────────────────────────────────────┤\n");
    printf("│ %-55s │\n", 
           convergence ? "🎉 Convergencia alcanzada" : "🛑 Límite de épocas alcanzado");
    printf("└─────────────────────────────────────────────────────────┘\n");
}

// Implementación REAL de MLP básico
void train_mlp_basico(Model *model, TrainingData data[], int num_samples) {
    printf("┌─────────────────────────────────────────────────────────┐\n");
    printf("│ 🕸️  ENTRENANDO MLP BÁSICO (2 CAPAS)                   │\n");
    printf("├─────────────────────────────────────────────────────────┤\n");
    printf("│ ℹ️  Puede resolver problemas no lineales como XOR       │\n");
    printf("└─────────────────────────────────────────────────────────┘\n\n");
    
    for(int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        double total_error = 0.0;
        
        for(int i = 0; i < num_samples; i++) {
            // Forward pass
            double output = forward_pass_mlp(&model->model.mlp_basico, data[i].features);
            double error = data[i].target - output;
            
            // Backpropagation
            backpropagate_mlp(&model->model.mlp_basico, data[i].features, data[i].target, model->learning_rate);
            
            total_error += error * error;
        }
        
        total_error /= num_samples;
        model->error_history[epoch] = total_error;
        model->epochs_trained = epoch + 1;
        
        if(epoch % 10 == 0 || epoch == MAX_EPOCHS - 1) {
            show_training_dashboard(model, data, num_samples, epoch, total_error);
            usleep(200000);
        }
        
        if(total_error < 0.01) break;
    }
    
    model->last_trained = time(NULL);
    
    printf("┌─────────────────────────────────────────────────────────┐\n");
    printf("│ ⚠️  ENTRENAMIENTO COMPLETADO                           │\n");
    printf("├─────────────────────────────────────────────────────────┤\n");
    printf("│ %-55s │\n", "🔄 Proceso finalizado");
    printf("└─────────────────────────────────────────────────────────┘\n");
}

void train_knn(Model *model, TrainingData data[], int num_samples) {
    printf("┌─────────────────────────────────────────────────────────┐\n");
    printf("│ 📍 ENTRENANDO K-VECINOS MÁS CERCANOS                  │\n");
    printf("├─────────────────────────────────────────────────────────┤\n");
    printf("│ ℹ️  Almacena los datos y los usa para clasificación    │\n");
    printf("└─────────────────────────────────────────────────────────┘\n\n");
    
    model->model.knn_model.num_samples = num_samples;
    for(int i = 0; i < num_samples; i++) {
        for(int j = 0; j < model->num_features; j++) {
            model->model.knn_model.samples[i][j] = data[i].features[j];
        }
        model->model.knn_model.targets[i] = data[i].target;
    }
    model->epochs_trained = 1;
    model->last_trained = time(NULL);
}

// Funciones de MLP básico
double forward_pass_mlp(const MLPBasico *mlp, const double features[]) {
    double hidden[5];
    
    // Capa entrada -> oculta
    for(int j = 0; j < mlp->num_hidden; j++) {
        hidden[j] = mlp->bias_layer1[j];
        for(int i = 0; i < mlp->num_features; i++) {
            hidden[j] += mlp->weights_layer1[i][j] * features[i];
        }
        hidden[j] = sigmoid(hidden[j]);
    }
    
    // Capa oculta -> salida
    double output = mlp->bias_layer2;
    for(int j = 0; j < mlp->num_hidden; j++) {
        output += mlp->weights_layer2[j] * hidden[j];
    }
    
    return sigmoid(output);
}

void backpropagate_mlp(MLPBasico *mlp, const double features[], int target, double learning_rate) {
    double hidden[5];
    double hidden_activation[5];
    
    // Forward pass
    for(int j = 0; j < mlp->num_hidden; j++) {
        hidden[j] = mlp->bias_layer1[j];
        for(int i = 0; i < mlp->num_features; i++) {
            hidden[j] += mlp->weights_layer1[i][j] * features[i];
        }
        hidden_activation[j] = sigmoid(hidden[j]);
    }
    
    double output = mlp->bias_layer2;
    for(int j = 0; j < mlp->num_hidden; j++) {
        output += mlp->weights_layer2[j] * hidden_activation[j];
    }
    double output_activation = sigmoid(output);
    
    // Backward pass
    double output_error = target - output_activation;
    double output_delta = output_error * output_activation * (1 - output_activation);
    
    // Actualizar pesos capa salida
    mlp->bias_layer2 += learning_rate * output_delta;
    for(int j = 0; j < mlp->num_hidden; j++) {
        mlp->weights_layer2[j] += learning_rate * output_delta * hidden_activation[j];
    }
    
    // Backpropagate a capa oculta
    for(int j = 0; j < mlp->num_hidden; j++) {
        double hidden_error = output_delta * mlp->weights_layer2[j];
        double hidden_delta = hidden_error * hidden_activation[j] * (1 - hidden_activation[j]);
        
        // Actualizar pesos capa oculta
        mlp->bias_layer1[j] += learning_rate * hidden_delta;
        for(int i = 0; i < mlp->num_features; i++) {
            mlp->weights_layer1[i][j] += learning_rate * hidden_delta * features[i];
        }
    }
}

// Funciones de predicción
int predict(const Model *model, const double features[]) {
    switch(model->algorithm) {
        case ALGO_PERCEPTRON_SIMPLE: {
            double z = model->model.perceptron.bias;
            for(int i = 0; i < model->num_features; i++) {
                z += model->model.perceptron.weights[i] * features[i];
            }
            double prediction = activation_function(model->model.perceptron.activation, z);
            return prediction > 0.5 ? 1 : 0;
        }
        
        case ALGO_MLP_BASICO: {
            double prediction = forward_pass_mlp(&model->model.mlp_basico, features);
            return prediction > 0.5 ? 1 : 0;
        }
        
        case ALGO_KNN: {
            if(model->model.knn_model.num_samples == 0) return 0;
            
            double distances[MAX_SAMPLES];
            for(int i = 0; i < model->model.knn_model.num_samples; i++) {
                double distance = 0;
                for(int j = 0; j < model->num_features; j++) {
                    double diff = features[j] - model->model.knn_model.samples[i][j];
                    distance += diff * diff;
                }
                distances[i] = sqrt(distance);
            }
            
            int class0 = 0, class1 = 0;
            int k = model->model.knn_model.k;
            if(k > model->model.knn_model.num_samples) {
                k = model->model.knn_model.num_samples;
            }
            
            for(int n = 0; n < k; n++) {
                int min_idx = -1;
                double min_dist = 1e10;
                for(int i = 0; i < model->model.knn_model.num_samples; i++) {
                    if(distances[i] < min_dist) {
                        min_dist = distances[i];
                        min_idx = i;
                    }
                }
                
                if(min_idx != -1) {
                    if(model->model.knn_model.targets[min_idx] == 0) class0++;
                    else class1++;
                    distances[min_idx] = 1e10;
                }
            }
            
            return class1 > class0 ? 1 : 0;
        }
        
        default:
            return 0;
    }
}

double predict_proba(const Model *model, const double features[]) {
    switch(model->algorithm) {
        case ALGO_PERCEPTRON_SIMPLE: {
            double z = model->model.perceptron.bias;
            for(int i = 0; i < model->num_features; i++) {
                z += model->model.perceptron.weights[i] * features[i];
            }
            return sigmoid(z);
        }
        
        case ALGO_MLP_BASICO: {
            return forward_pass_mlp(&model->model.mlp_basico, features);
        }
        
        case ALGO_KNN: {
            if(model->model.knn_model.num_samples == 0) return 0.0;
            
            double distances[MAX_SAMPLES];
            for(int i = 0; i < model->model.knn_model.num_samples; i++) {
                double distance = 0;
                for(int j = 0; j < model->num_features; j++) {
                    double diff = features[j] - model->model.knn_model.samples[i][j];
                    distance += diff * diff;
                }
                distances[i] = sqrt(distance);
            }
            
            int class1_count = 0;
            int k = model->model.knn_model.k;
            if(k > model->model.knn_model.num_samples) {
                k = model->model.knn_model.num_samples;
            }
            
            for(int n = 0; n < k; n++) {
                int min_idx = -1;
                double min_dist = 1e10;
                for(int i = 0; i < model->model.knn_model.num_samples; i++) {
                    if(distances[i] < min_dist) {
                        min_dist = distances[i];
                        min_idx = i;
                    }
                }
                
                if(min_idx != -1) {
                    if(model->model.knn_model.targets[min_idx] == 1) class1_count++;
                    distances[min_idx] = 1e10;
                }
            }
            
            return (double)class1_count / k;
        }
        
        default:
            return predict(model, features) ? 1.0 : 0.0;
    }
}

// Funciones de activación
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double tanh_activation(double x) {
    return tanh(x);
}

double relu(double x) {
    return x > 0 ? x : 0;
}

double step_function(double x) {
    return x >= 0 ? 1.0 : 0.0;
}

double activation_function(ActivationFunction func, double x) {
    switch(func) {
        case ACTIVATION_SIGMOIDE: return sigmoid(x);
        case ACTIVATION_TANH: return tanh_activation(x);
        case ACTIVATION_RELU: return relu(x);
        case ACTIVATION_STEP: return step_function(x);
        default: return sigmoid(x);
    }
}

double activation_derivative(ActivationFunction func, double x) {
    switch(func) {
        case ACTIVATION_SIGMOIDE: {
            double s = sigmoid(x);
            return s * (1 - s);
        }
        case ACTIVATION_TANH: {
            double t = tanh(x);
            return 1 - t * t;
        }
        case ACTIVATION_RELU: return x > 0 ? 1.0 : 0.0;
        case ACTIVATION_STEP: return 0.0;
        default: {
            double s = sigmoid(x);
            return s * (1 - s);
        }
    }
}

// Funciones auxiliares
int load_training_data(const char *filename, TrainingData data[], int max_samples) {
    FILE *file = fopen(filename, "r");
    if(!file) return 0;
    
    char line[256];
    int count = 0;
    
    while(fgets(line, sizeof(line), file) && count < max_samples) {
        if(line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        
        double x1, x2;
        int target;
        if(sscanf(line, "%lf %lf %d", &x1, &x2, &target) == 3) {
            data[count].features[0] = x1;
            data[count].features[1] = x2;
            data[count].target = target;
            sprintf(data[count].label, "Clase %d", target);
            count++;
        }
    }
    
    fclose(file);
    return count;
}

void clear_screen() {
    printf("\033[2J\033[1;1H");
}

void print_help() {
    printf("┌─────────────────────────────────────────────────────────┐\n");
    printf("│ 🎯 SISTEMA DE MACHINE LEARNING - AYUDA                │\n");
    printf("├─────────────────────────────────────────────────────────┤\n");
    printf("│ ⚠️  ALGORITMOS IMPLEMENTADOS:                         │\n");
    printf("│   - Perceptrón Simple: Problemas LINEALES (AND, OR)    │\n");
    printf("│   - MLP Básico: Problemas NO LINEALES (XOR)            │\n");
    printf("│   - K-Vecinos: Algoritmo basado en instancias          │\n");
    printf("├─────────────────────────────────────────────────────────┤\n");
    printf("│ 🔥 MODO AUTOMÁTICO (NUEVO):                           │\n");
    printf("│   ./programa datos.txt                                 │\n");
    printf("│   - Carga y entrena automáticamente                    │\n");
    printf("│   - Selecciona el mejor algoritmo                      │\n");
    printf("│   - Permite predicciones interactivas                  │\n");
    printf("├─────────────────────────────────────────────────────────┤\n");
    printf("│ 🧠 ENTRENAR modelo:                                   │\n");
    printf("│   ./programa --entrenar --problema AND|OR|XOR          │\n");
    printf("│   ./programa --entrenar datos.txt                      │\n");
    printf("├─────────────────────────────────────────────────────────┤\n");
    printf("│ 🎯 USAR modelo:                                       │\n");
    printf("│   ./programa --usar modelo.bin [datos.txt]             │\n");
    printf("├─────────────────────────────────────────────────────────┤\n");
    printf("│ 🎨 VISUALIZAR modelo:                                 │\n");
    printf("│   ./programa --visualizar modelo.bin [datos.txt]       │\n");
    printf("├─────────────────────────────────────────────────────────┤\n");
    printf("│ 📊 COMPARAR modelos:                                  │\n");
    printf("│   ./programa --comparar modelo1.bin modelo2.bin ...    │\n");
    printf("├─────────────────────────────────────────────────────────┤\n");
    printf("│ 📚 PROBLEMAS PREDEFINIDOS:                            │\n");
    printf("│   ./programa --problemas                               │\n");
    printf("├─────────────────────────────────────────────────────────┤\n");
    printf("│ Ejemplos prácticos:                                    │\n");
    printf("│   ./programa mis_datos.txt                             │\n");
    printf("│   ./programa --entrenar --problema XOR --algoritmo mlp │\n");
    printf("│   ./programa --usar modelo.bin                         │\n");
    printf("└─────────────────────────────────────────────────────────┘\n");
}

// NUEVA FUNCIÓN: Modo automático que entrena y usa el modelo
void automatic_mode(const char *data_filename) {
    TrainingData data[MAX_SAMPLES];
    int num_samples = load_training_data(data_filename, data, MAX_SAMPLES);
    
    if(num_samples == 0) {
        printf("┌─────────────────────────────────────────────────────────┐\n");
        printf("│ ❌ ERROR CARGANDO DATOS                               │\n");
        printf("├─────────────────────────────────────────────────────────┤\n");
        printf("│ No se pudo cargar: %-35s │\n", data_filename);
        printf("│ Formato esperado: x1 x2 target                         │\n");
        printf("│ Ejemplo:                                               │\n");
        printf("│   0.0 0.0 0                                            │\n");
        printf("│   0.0 1.0 1                                            │\n");
        printf("│   1.0 0.0 1                                            │\n");
        printf("│   1.0 1.0 0                                            │\n");
        printf("└─────────────────────────────────────────────────────────┘\n");
        return;
    }
    
    printf("┌─────────────────────────────────────────────────────────┐\n");
    printf("│ ✅ DATOS CARGADOS EXITOSAMENTE                         │\n");
    printf("├─────────────────────────────────────────────────────────┤\n");
    printf("│ Muestras: %-45d │\n", num_samples);
    printf("└─────────────────────────────────────────────────────────┘\n\n");
    
    // Mostrar análisis de datos
    show_data_analysis(data, num_samples, 2);
    
    // Seleccionar algoritmo automáticamente basado en los datos
    AlgorithmType algo;
    printf("┌─────────────────────────────────────────────────────────┐\n");
    printf("│ 🔍 ANALIZANDO DATOS                                    │\n");
    printf("├─────────────────────────────────────────────────────────┤\n");
    
    // Detectar si es un problema lineal o no lineal
    int is_linear = 1;
    for(int i = 0; i < num_samples && is_linear; i++) {
        for(int j = i + 1; j < num_samples && is_linear; j++) {
            if(data[i].target != data[j].target) {
                double dx = data[i].features[0] - data[j].features[0];
                double dy = data[i].features[1] - data[j].features[1];
                if(fabs(dx) > 0.5 && fabs(dy) > 0.5) {
                    is_linear = 0;
                }
            }
        }
    }
    
    if(is_linear) {
        algo = ALGO_PERCEPTRON_SIMPLE;
        printf("│ ✅ Problema lineal detectado                          │\n");
        printf("│ 🎯 Usando Perceptrón Simple                          │\n");
    } else {
        algo = ALGO_MLP_BASICO;
        printf("│ ✅ Problema no lineal detectado                       │\n");
        printf("│ 🎯 Usando MLP Básico                                 │\n");
    }
    printf("└─────────────────────────────────────────────────────────┘\n");
    
    // Entrenar el modelo
    Model model;
    initialize_model(&model, algo, 2);
    
    printf("\n");
    print_model_info(&model);
    
    printf("┌─────────────────────────────────────────────────────────┐\n");
    printf("│ 🚀 INICIANDO ENTRENAMIENTO                            │\n");
    printf("└─────────────────────────────────────────────────────────┘\n\n");
    
    switch(algo) {
        case ALGO_PERCEPTRON_SIMPLE:
            train_perceptron_simple(&model, data, num_samples);
            break;
        case ALGO_MLP_BASICO:
            train_mlp_basico(&model, data, num_samples);
            break;
        case ALGO_KNN:
            train_knn(&model, data, num_samples);
            break;
    }
    
    // Mostrar resultados del entrenamiento
    double accuracy = calculate_accuracy(&model, data, num_samples);
    printf("┌─────────────────────────────────────────────────────────┐\n");
    printf("│ 🎯 RESULTADOS FINALES                                 │\n");
    printf("├─────────────────────────────────────────────────────────┤\n");
    printf("│ Precisión: %-43.1f%% │\n", accuracy * 100);
    printf("└─────────────────────────────────────────────────────────┘\n");
    
    // Guardar el modelo automáticamente
    char model_filename[100];
    snprintf(model_filename, sizeof(model_filename), "modelo_automatico_%ld.bin", time(NULL));
    if(save_model(&model, model_filename)) {
        printf("┌─────────────────────────────────────────────────────────┐\n");
        printf("│ 💾 MODELO GUARDADO                                    │\n");
        printf("├─────────────────────────────────────────────────────────┤\n");
        printf("│ Archivo: %-45s │\n", model_filename);
        printf("└─────────────────────────────────────────────────────────┘\n");
    }
    
    // Modo interactivo para hacer predicciones
    printf("┌─────────────────────────────────────────────────────────┐\n");
    printf("│ 🎮 MODO INTERACTIVO DE PREDICCIONES                   │\n");
    printf("├─────────────────────────────────────────────────────────┤\n");
    printf("│ Ingrese datos (x1 x2) o 'fin' para terminar             │\n");
    printf("└─────────────────────────────────────────────────────────┘\n\n");
    
    char input[100];
    while(1) {
        printf("📥 Ingrese datos para predecir: ");
        if(fgets(input, sizeof(input), stdin) == NULL) break;
        
        if(strncmp(input, "fin", 3) == 0) break;
        
        double x1, x2;
        if(sscanf(input, "%lf %lf", &x1, &x2) == 2) {
            double features[2] = {x1, x2};
            int prediction = predict(&model, features);
            double confidence = predict_proba(&model, features);
            
            printf("   📊 Predicción: %d (Confianza: %.1f%%)\n", 
                   prediction, confidence * 100);
            
            // Mostrar interpretación
            if(prediction == 0) {
                printf("   🔍 Interpretación: Clase 0");
            } else {
                printf("   🔍 Interpretación: Clase 1");
            }
            
            if(confidence > 0.8) {
                printf(" - ✅ Alta confianza\n");
            } else if(confidence > 0.6) {
                printf(" - ⚠️  Confianza media\n");
            } else {
                printf(" - ❗ Baja confianza\n");
            }
        } else {
            printf("❌ Formato incorrecto. Use: x1 x2\n");
        }
    }
    
    printf("\n┌─────────────────────────────────────────────────────────┐\n");
    printf("│ ✅ PROGRAMA TERMINADO                                  │\n");
    printf("├─────────────────────────────────────────────────────────┤\n");
    printf("│ Modelo guardado en: %-35s │\n", model_filename);
    printf("│ Reutilizar con: ./programa --usar %-19s │\n", model_filename);
    printf("└─────────────────────────────────────────────────────────┘\n");
}

int main(int argc, char *argv[]) {
    srand(time(NULL));
    
    if(argc < 2) {
        print_help();
        return 1;
    }

    // MODO AUTOMÁTICO: Si el argumento es un archivo de datos
    if(argv[1][0] != '-') {
        printf("┌─────────────────────────────────────────────────────────┐\n");
        printf("│ 🤖 MODO AUTOMÁTICO ACTIVADO                           │\n");
        printf("├─────────────────────────────────────────────────────────┤\n");
        printf("│ Archivo de datos: %-35s │\n", argv[1]);
        printf("│ Entrenamiento automático iniciado                      │\n");
        printf("└─────────────────────────────────────────────────────────┘\n\n");
        
        automatic_mode(argv[1]);
        return 0;
    }
    
    // Modos de operación con flags
    if(strcmp(argv[1], "--entrenar") == 0) {
        TrainingData data[MAX_SAMPLES];
        int num_samples = 0;
        char model_filename[100] = "modelo.bin";
        AlgorithmType algo = ALGO_PERCEPTRON_SIMPLE;
        int use_predefined = 0;
        
        // Procesar argumentos
        for(int i = 2; i < argc; i++) {
            if(strcmp(argv[i], "--algoritmo") == 0 && i+1 < argc) {
                i++;
                if(strcmp(argv[i], "perceptron") == 0) algo = ALGO_PERCEPTRON_SIMPLE;
                else if(strcmp(argv[i], "mlp") == 0) algo = ALGO_MLP_BASICO;
                else if(strcmp(argv[i], "knn") == 0) algo = ALGO_KNN;
            }
            else if(strcmp(argv[i], "--problema") == 0 && i+1 < argc) {
                i++;
                use_predefined = 1;
                if(strcmp(argv[i], "AND") == 0) generate_and_data(data, &num_samples);
                else if(strcmp(argv[i], "OR") == 0) generate_or_data(data, &num_samples);
                else if(strcmp(argv[i], "XOR") == 0) generate_xor_data(data, &num_samples);
                else {
                    printf("❌ Problema desconocido. Usando XOR por defecto.\n");
                    generate_xor_data(data, &num_samples);
                }
            }
            else if(strcmp(argv[i], "--modelo") == 0 && i+1 < argc) {
                i++;
                strcpy(model_filename, argv[i]);
            }
            else {
                // Asumir que es archivo de datos
                num_samples = load_training_data(argv[i], data, MAX_SAMPLES);
                if(num_samples == 0) {
                    printf("❌ No se pudo cargar %s\n", argv[i]);
                    return 1;
                }
            }
        }
        
        // Si no se especificó problema ni archivo, mostrar menú
        if(num_samples == 0 && !use_predefined) {
            show_problem_menu();
            return 1;
        }
        
        printf("┌─────────────────────────────────────────────────────────┐\n");
        printf("│ 🎯 ENTRENAMIENTO DE MODELO                           │\n");
        printf("└─────────────────────────────────────────────────────────┘\n\n");
        
        Model model;
        initialize_model(&model, algo, 2);
        
        show_data_analysis(data, num_samples, 2);
        printf("\n");
        print_model_info(&model);
        
        printf("┌─────────────────────────────────────────────────────────┐\n");
        printf("│ 🚀 INICIANDO ENTRENAMIENTO                            │\n");
        printf("└─────────────────────────────────────────────────────────┘\n\n");
        
        // Entrenar según el algoritmo seleccionado
        switch(algo) {
            case ALGO_PERCEPTRON_SIMPLE:
                train_perceptron_simple(&model, data, num_samples);
                break;
            case ALGO_MLP_BASICO:
                train_mlp_basico(&model, data, num_samples);
                break;
            case ALGO_KNN:
                train_knn(&model, data, num_samples);
                break;
        }
        
        // Guardar modelo entrenado
        if(save_model(&model, model_filename)) {
            printf("┌─────────────────────────────────────────────────────────┐\n");
            printf("│ 💾 MODELO GUARDADO                                    │\n");
            printf("├─────────────────────────────────────────────────────────┤\n");
            printf("│ Archivo: %-45s │\n", model_filename);
            printf("└─────────────────────────────────────────────────────────┘\n");
            
            // Mostrar evaluación final
            double accuracy = calculate_accuracy(&model, data, num_samples);
            printf("┌─────────────────────────────────────────────────────────┐\n");
            printf("│ 📊 EVALUACIÓN FINAL                                   │\n");
            printf("├─────────────────────────────────────────────────────────┤\n");
            printf("│ Precisión: %-43.1f%% │\n", accuracy * 100);
            printf("└─────────────────────────────────────────────────────────┘\n");
            
            print_confusion_matrix(&model, data, num_samples);
        } else {
            printf("❌ Error guardando el modelo\n");
        }
        
    } else if(strcmp(argv[1], "--usar") == 0 && argc >= 3) {
        char model_filename[100] = "modelo.bin";
        char data_filename[100] = "";
        
        // Procesar argumentos
        for(int i = 2; i < argc; i++) {
            if(strcmp(argv[i], "--modelo") == 0 && i+1 < argc) {
                i++;
                strcpy(model_filename, argv[i]);
            } else {
                strcpy(data_filename, argv[i]);
            }
        }
        
        Model model;
        if(!load_model(&model, model_filename)) {
            printf("❌ Error cargando modelo %s\n", model_filename);
            return 1;
        }
        
        TrainingData new_data[MAX_SAMPLES];
        int num_samples = 0;
        
        if(strlen(data_filename) > 0) {
            num_samples = load_training_data(data_filename, new_data, MAX_SAMPLES);
        } else {
            // Si no hay archivo de datos, usar entrada manual
            printf("┌─────────────────────────────────────────────────────────┐\n");
            printf("│ 📝 ENTRADA MANUAL DE DATOS                           │\n");
            printf("├─────────────────────────────────────────────────────────┤\n");
            printf("│ Formato: x1 x2                                        │\n");
            printf("│ Ejemplo: 0 0 para clase 0, 1 1 para clase 1           │\n");
            printf("│ Escriba 'fin' para terminar                           │\n");
            printf("└─────────────────────────────────────────────────────────┘\n");
            
            char input[100];
            while(num_samples < MAX_SAMPLES) {
                printf("📥 Datos %d: ", num_samples + 1);
                if(fgets(input, sizeof(input), stdin) == NULL) break;
                
                if(strncmp(input, "fin", 3) == 0) break;
                
                double x1, x2;
                if(sscanf(input, "%lf %lf", &x1, &x2) == 2) {
                    new_data[num_samples].features[0] = x1;
                    new_data[num_samples].features[1] = x2;
                    new_data[num_samples].target = -1; // Desconocido para predicción
                    num_samples++;
                } else {
                    printf("❌ Formato incorrecto. Use: x1 x2\n");
                }
            }
        }
        
        printf("┌─────────────────────────────────────────────────────────┐\n");
        printf("│ 🎯 USANDO MODELO ENTRENADO                           │\n");
        printf("└─────────────────────────────────────────────────────────┘\n\n");
        
        print_model_info(&model);
        
        // Realizar predicciones
        printf("┌─────────────────────────────────────────────────────────┐\n");
        printf("│ 📊 PREDICCIONES                                       │\n");
        printf("└─────────────────────────────────────────────────────────┘\n");
        
        for(int i = 0; i < num_samples; i++) {
            int prediction = predict(&model, new_data[i].features);
            double confidence = predict_proba(&model, new_data[i].features);
            
            printf("   Muestra %d: [%.1f, %.1f] → ", i+1, 
                   new_data[i].features[0], new_data[i].features[1]);
            printf("Predicción: %d", prediction);
            printf(" (Confianza: %.1f%%)", confidence * 100);
            
            if(new_data[i].target != -1) {
                if(prediction == new_data[i].target) {
                    printf(" ✅");
                } else {
                    printf(" ❌");
                }
            }
            printf("\n");
        }
        
    } else if(strcmp(argv[1], "--visualizar") == 0) {
        char model_filename[100] = "modelo.bin";
        char data_filename[100] = "";
        
        // Procesar argumentos
        for(int i = 2; i < argc; i++) {
            if(strcmp(argv[i], "--modelo") == 0 && i+1 < argc) {
                i++;
                strcpy(model_filename, argv[i]);
            } else {
                strcpy(data_filename, argv[i]);
            }
        }
        
        Model model;
        if(!load_model(&model, model_filename)) {
            printf("❌ Error cargando modelo %s\n", model_filename);
            return 1;
        }
        
        TrainingData data[MAX_SAMPLES];
        int num_samples = 0;
        
        if(strlen(data_filename) > 0) {
            num_samples = load_training_data(data_filename, data, MAX_SAMPLES);
        } else {
            // Generar datos del problema XOR para visualización
            generate_xor_data(data, &num_samples);
        }
        
        printf("┌─────────────────────────────────────────────────────────┐\n");
        printf("│ 🎨 VISUALIZACIÓN AVANZADA                            │\n");
        printf("└─────────────────────────────────────────────────────────┘\n\n");
        
        print_model_info(&model);
        printf("\n");
        show_confidence_map_enhanced(&model, data, num_samples);
        printf("\n");
        print_confusion_matrix(&model, data, num_samples);
        
    } else if(strcmp(argv[1], "--comparar") == 0 && argc >= 3) {
        // Modo comparación
        printf("┌─────────────────────────────────────────────────────────┐\n");
        printf("│ 📊 COMPARACIÓN DE MODELOS                             │\n");
        printf("└─────────────────────────────────────────────────────────┘\n\n");
        
        Model *models[MAX_MODELS];
        int num_models = argc - 2;
        
        if(num_models > MAX_MODELS) {
            printf("❌ Demasiados modelos (máximo %d)\n", MAX_MODELS);
            return 1;
        }
        
        // Cargar modelos
        for(int i = 0; i < num_models; i++) {
            models[i] = malloc(sizeof(Model));
            if(!load_model(models[i], argv[2 + i])) {
                printf("❌ Error cargando modelo: %s\n", argv[2 + i]);
                return 1;
            }
        }
        
        // Cargar datos de prueba si se proporcionan
        TrainingData test_data[MAX_SAMPLES];
        int num_test_samples = 0;
        
        if(argc >= 3 + num_models) {
            num_test_samples = load_training_data(argv[2 + num_models], test_data, MAX_SAMPLES);
        }
        
        if(num_test_samples > 0) {
            show_comparison_metrics(models, num_models, test_data, num_test_samples);
        } else {
            printf("ℹ️  Proporciona archivo de datos para comparar métricas\n");
        }
        
        // Liberar memoria
        for(int i = 0; i < num_models; i++) {
            free(models[i]);
        }
        
    } else if(strcmp(argv[1], "--problemas") == 0) {
        show_problem_menu();
        
    } else {
        print_help();
    }
    
    return 0;
}
