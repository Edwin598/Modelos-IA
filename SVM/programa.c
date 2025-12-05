/*******************************************************************************
 * SISTEMA SVM DIDACTICO COMPLETO - SUPPORT VECTOR MACHINES
 * Sistema educativo completo para aprender SVM desde cero
 * Caracteristicas:
 * - SVM con kernels lineal, RBF y polinomial
 * - Entrenamiento visual paso a paso
 * - Explicaciones detalladas de cada concepto
 * - Modo "aprendizaje activo" con preguntas y respuestas
 * - Simulacion de diferentes tipos de datos
 * - Analisis de errores y sobreajuste
 * - Exportacion de informes completos
 * - Persistencia de modelos y datasets
 * - Sistema de ayuda contextual
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
#include <unistd.h>
#include <stdbool.h>
#include <float.h>
#include <stdarg.h>  // Para va_start, va_end
#include <errno.h>   // Para manejo de errores

// ============================ CONFIGURACION ============================
#define MAX_SAMPLES 1000
#define MAX_FEATURES 10
#define MAX_CLASSES 2
#define MAX_ITERATIONS 1000
#define LEARNING_RATE 0.01
#define TERMINAL_WIDTH 80
#define TERMINAL_HEIGHT 24
#define COLOR_RESET "\033[0m"
#define COLOR_RED "\033[31m"
#define COLOR_GREEN "\033[32m"
#define COLOR_YELLOW "\033[33m"
#define COLOR_BLUE "\033[34m"
#define COLOR_MAGENTA "\033[35m"
#define COLOR_CYAN "\033[36m"
#define COLOR_WHITE "\033[37m"

// ============================ ESTRUCTURAS DE DATOS ============================

typedef struct {
    double features[MAX_FEATURES];
    int label;  // SVM usa -1 o +1
    double alpha;  // Multiplicador de Lagrange
    int is_support_vector;
    int is_misclassified;  // Para analisis de errores
} DataPoint;

typedef struct {
    DataPoint points[MAX_SAMPLES];
    int num_samples;
    int num_features;
    char feature_names[MAX_FEATURES][50];
    double feature_min[MAX_FEATURES];
    double feature_max[MAX_FEATURES];
    int is_normalized;
    char name[100];
    char description[256];
} Dataset;

typedef struct {
    double weights[MAX_FEATURES];
    double bias;
    double C;
    double gamma;
    double degree;  // Para kernel polinomial
    char kernel_type[20];
    DataPoint support_vectors[MAX_SAMPLES];
    int num_support_vectors;
    double margin;
    double accuracy;
    double training_accuracy;
    double validation_accuracy;
    int iterations;
    time_t trained_at;
    char name[100];
    int num_features_trained;
    double training_time;  // Tiempo de entrenamiento en segundos
} SVM_Model;

typedef struct {
    double error_history[MAX_ITERATIONS];
    double margin_history[MAX_ITERATIONS];
    int sv_count_history[MAX_ITERATIONS];
    double accuracy_history[MAX_ITERATIONS];
    int iteration_count;
} TrainingHistory;

typedef struct {
    double train_accuracy;
    double test_accuracy;
    double precision;
    double recall;
    double f1_score;
    double hinge_loss;
    int true_positives;
    int true_negatives;
    int false_positives;
    int false_negatives;
} ModelMetrics;

typedef struct {
    char question[256];
    char options[4][100];
    int correct_answer;
    char explanation[512];
} QuizQuestion;

// ============================ VARIABLES GLOBALES ============================
Dataset current_dataset = {0};
SVM_Model current_model = {0};
TrainingHistory training_history = {0};
ModelMetrics current_metrics = {0};
int terminal_width = TERMINAL_WIDTH;
char current_model_file[256] = "";
char current_dataset_file[256] = "";
int learning_mode = 0;  // 0=normal, 1=paso a paso, 2=examen
int quiz_score = 0;
int total_questions = 0;
QuizQuestion quiz_questions[20];  // Array de preguntas

// ============================ PROTOTIPOS DE FUNCIONES ============================

// Sistema e inicializacion
void init_system();
void cleanup_system();
void print_header(const char* title);
void print_separator(char ch);
void clear_screen();
void wait_for_key(const char* message);
void wait_for_enter();
int get_terminal_width();
void print_help();
void print_welcome_message();
void setup_learning_environment();

// Visualizacion mejorada
void print_progress_bar(int current, int total, const char* label, double value, int color);
void print_svm_visualization(Dataset* dataset, SVM_Model* model);
void print_decision_boundary_2d(Dataset* dataset, SVM_Model* model);
void print_support_vectors_visualization(SVM_Model* model);
void print_training_progress(SVM_Model* model, int iteration, double error, double accuracy);
void print_hyperplane_equation(SVM_Model* model, Dataset* dataset);
void print_margin_visualization(Dataset* dataset, SVM_Model* model);
void print_kernel_comparison();
void print_model_info(SVM_Model* model);
void print_dataset_visualization(Dataset* dataset);
void print_feature_importance(SVM_Model* model, Dataset* dataset);
void print_error_analysis(Dataset* dataset, SVM_Model* model);
void print_learning_curve(TrainingHistory* history);
void print_confusion_matrix_visual(int tp, int tn, int fp, int fn);

// Manejo de datasets mejorado
Dataset load_dataset(const char* filename);
Dataset create_linearly_separable_dataset(int samples);
Dataset create_xor_dataset(int samples);
Dataset create_circular_dataset(int samples);
Dataset create_spiral_dataset(int samples);
Dataset create_moon_dataset(int samples);
Dataset create_random_dataset(int samples, int features, int complexity);
void normalize_dataset(Dataset* dataset);
void print_dataset_info(Dataset* dataset);
void split_dataset(Dataset* dataset, Dataset* train, Dataset* test, double train_ratio);
void save_dataset(Dataset* dataset, const char* filename);
void augment_dataset(Dataset* dataset, int factor);
void add_noise_to_dataset(Dataset* dataset, double noise_level);

// Funciones SVM mejoradas
double dot_product(double a[], double b[], int n);
double linear_kernel(double a[], double b[], int n);
double rbf_kernel(double a[], double b[], int n, double gamma);
double polynomial_kernel(double a[], double b[], int n, double degree);
double predict_point(SVM_Model* model, DataPoint* point);
double compute_margin(SVM_Model* model);
int is_support_vector(DataPoint* point, SVM_Model* model, double tolerance);
double compute_distance_to_hyperplane(SVM_Model* model, DataPoint* point);

// Entrenamiento mejorado
void train_svm_linear(Dataset* dataset, SVM_Model* model);
void train_svm_sgd(Dataset* dataset, SVM_Model* model);
void train_svm_with_validation(Dataset* dataset, SVM_Model* model, double validation_split);
void train_svm_step_by_step(Dataset* dataset, SVM_Model* model);
double compute_hinge_loss(SVM_Model* model, Dataset* dataset);
double compute_regularization_term(SVM_Model* model);
void update_weights_sgd(SVM_Model* model, DataPoint* point, double learning_rate);

// Evaluacion mejorada
double evaluate_svm(SVM_Model* model, Dataset* dataset);
void confusion_matrix_svm(SVM_Model* model, Dataset* dataset, int matrix[2][2]);
void print_performance_metrics(SVM_Model* model, Dataset* dataset);
ModelMetrics compute_detailed_metrics(SVM_Model* model, Dataset* dataset);
void cross_validation(Dataset* dataset, SVM_Model* model, int folds);
void grid_search_hyperparameters(Dataset* dataset);

// Persistencia de modelos
int save_model(SVM_Model* model, const char* filename);
int load_model(SVM_Model* model, const char* filename);
void save_model_interactive(SVM_Model* model);
void load_model_interactive(SVM_Model* model);
void export_model_to_text(SVM_Model* model, const char* filename);
void export_full_report(SVM_Model* model, Dataset* dataset, const char* filename);

// Sistema de aprendizaje
void learning_mode_menu();
void interactive_tutorial();
void step_by_step_training();
void concept_explanation(const char* concept);
void take_quiz();
void show_quiz_results();
void ask_question(QuizQuestion* question);
void load_quiz_questions();
void explain_misconceptions(SVM_Model* model, Dataset* dataset);

// Analisis y diagnostico
void analyze_model_performance(SVM_Model* model, Dataset* dataset);
void diagnose_overfitting(SVM_Model* model, Dataset* train, Dataset* test);
void suggest_improvements(SVM_Model* model, Dataset* dataset);
void compare_models(SVM_Model* models[], int num_models, Dataset* dataset);

// Interfaz mejorada
void interactive_mode();
void training_mode();
void visualization_mode();
void demo_mode();
void tutorial_mode();
void interactive_prediction();
void batch_prediction();
void show_performance();
void show_kernel_comparison();
void show_hyperplane();
void model_management_menu();
void dataset_management_menu();
void analysis_mode();
void settings_mode();

// Utilidades
void print_color(const char* color, const char* format, ...);
void print_bullet(const char* text, int indent);
void print_section(const char* title);
void print_note(const char* note);
void print_warning(const char* warning);
void print_success(const char* message);
void print_error(const char* error);
void center_text(const char* text);
void draw_box(const char* title, const char* content);
void animate_progress(const char* message, int duration_ms);

// ============================ FUNCION PRINCIPAL ============================

int main(int argc, char* argv[]) {
    init_system();
    
    int interactive = 0;
    char* data_file = NULL;
    char* model_file = NULL;
    int demo = 0;
    int tutorial = 0;
    int learning = 0;
    
    // Parsear argumentos
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0) interactive = 1;
        else if (strcmp(argv[i], "-d") == 0 && i+1 < argc) data_file = argv[++i];
        else if (strcmp(argv[i], "-m") == 0 && i+1 < argc) model_file = argv[++i];
        else if (strcmp(argv[i], "-demo") == 0) demo = 1;
        else if (strcmp(argv[i], "-t") == 0) tutorial = 1;
        else if (strcmp(argv[i], "-learn") == 0) learning = 1;
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_help();
            cleanup_system();
            return 0;
        }
    }
    
    clear_screen();
    print_welcome_message();
    
    // Configurar modo aprendizaje si se solicita
    if (learning) {
        learning_mode = 1;
        setup_learning_environment();
    }
    
    // Cargar modelo si se especifico
    if (model_file) {
        print_color(COLOR_CYAN, "Cargando modelo: %s\n", model_file);
        if (load_model(&current_model, model_file)) {
            strcpy(current_model_file, model_file);
            print_success("Modelo cargado exitosamente!");
            print_model_info(&current_model);
            
            // Si hay dataset, evaluar el modelo
            if (data_file) {
                print_color(COLOR_CYAN, "\nCargando dataset: %s\n", data_file);
                current_dataset = load_dataset(data_file);
                if (current_dataset.num_samples > 0) {
                    strcpy(current_dataset_file, data_file);
                    normalize_dataset(&current_dataset);
                    evaluate_svm(&current_model, &current_dataset);
                }
            }
        } else {
            print_error("No se pudo cargar el modelo. Inicializando nuevo modelo.");
        }
    }
    
    // Cargar dataset si se especifico
    if (data_file && !model_file) {
        print_color(COLOR_CYAN, "Cargando dataset: %s\n", data_file);
        current_dataset = load_dataset(data_file);
        if (current_dataset.num_samples == 0) {
            print_warning("Error al cargar. Generando dataset de ejemplo.");
            current_dataset = create_linearly_separable_dataset(100);
            strcpy(current_dataset.name, "Dataset Lineal de Ejemplo");
        } else {
            strcpy(current_dataset_file, data_file);
        }
    } else if (!model_file) {
        print_color(COLOR_CYAN, "Generando dataset de ejemplo...\n");
        current_dataset = create_linearly_separable_dataset(100);
        strcpy(current_dataset.name, "Dataset Lineal de Ejemplo");
    }
    
    // Normalizar dataset si existe
    if (current_dataset.num_samples > 0) {
        normalize_dataset(&current_dataset);
        print_dataset_info(&current_dataset);
    }
    
    // Ejecutar modo apropiado
    if (tutorial) {
        tutorial_mode();
    } else if (demo) {
        demo_mode();
    } else if (learning || interactive || argc == 1) {
        if (learning) {
            learning_mode_menu();
        } else {
            interactive_mode();
        }
    }
    
    cleanup_system();
    return 0;
}

// ============================ IMPLEMENTACIONES ============================

void init_system() {
    printf("\033[2J\033[1;1H");  // Clear screen
    print_color(COLOR_CYAN, "🚀 Inicializando Sistema SVM Didactico...\n");
    
    srand(time(NULL));
    terminal_width = TERMINAL_WIDTH;
    
    // Inicializar variables globales
    memset(current_model_file, 0, sizeof(current_model_file));
    memset(current_dataset_file, 0, sizeof(current_dataset_file));
    learning_mode = 0;
    quiz_score = 0;
    total_questions = 0;
    
    // Cargar preguntas del quiz
    load_quiz_questions();
    
    // Configurar salida
    setbuf(stdout, NULL);
    
    print_success("Sistema inicializado correctamente.\n");
    sleep(1);
}

void cleanup_system() {
    printf("\n");
    print_separator('=');
    print_color(COLOR_YELLOW, "🧹 Finalizando Sistema SVM Didactico\n");
    
    // Mostrar resumen si hay modelo entrenado
    if (strlen(current_model.kernel_type) > 0) {
        printf("\nResumen de la sesion:\n");
        printf("  • Modelo: %s\n", current_model.name);
        printf("  • Kernel: %s\n", current_model.kernel_type);
        printf("  • Exactitud: %.2f%%\n", current_model.accuracy * 100);
        
        if (learning_mode) {
            printf("  • Puntaje del quiz: %d/%d\n", quiz_score, total_questions);
        }
    }
    
    printf("\n¡Gracias por usar el Sistema SVM Didactico!\n");
    print_separator('=');
}

void print_header(const char* title) {
    printf("\n");
    print_separator('=');
    center_text(title);
    print_separator('=');
}

void print_separator(char ch) {
    for (int i = 0; i < terminal_width; i++) printf("%c", ch);
    printf("\n");
}

void clear_screen() {
    printf("\033[2J\033[1;1H");
}

void wait_for_key(const char* message) {
    if (message) printf("\n%s", message);
    print_color(COLOR_YELLOW, " (Presione Enter para continuar...)");
    fflush(stdout);
    getchar();
}

void wait_for_enter() {
    printf("\n");
    print_color(COLOR_YELLOW, "Presione Enter para continuar...");
    fflush(stdout);
    getchar();
}

int get_terminal_width() {
    return TERMINAL_WIDTH;
}

void print_help() {
    print_header("AYUDA DEL SISTEMA SVM DIDACTICO");
    
    printf("\nUso: programa [opciones]\n\n");
    printf("Opciones:\n");
    printf("  -i            Modo interactivo (por defecto)\n");
    printf("  -d ARCHIVO    Cargar dataset desde archivo CSV\n");
    printf("  -m ARCHIVO    Cargar modelo entrenado\n");
    printf("  -demo         Ejecutar demostracion automatica\n");
    printf("  -t            Modo tutorial paso a paso\n");
    printf("  -learn        Modo aprendizaje activo\n");
    printf("  -h, --help    Mostrar esta ayuda\n");
    
    printf("\nEjemplos:\n");
    printf("  programa -d datos.csv          # Cargar dataset y entrenar\n");
    printf("  programa -m modelo.svm         # Cargar modelo existente\n");
    printf("  programa -learn                # Modo aprendizaje guiado\n");
    printf("  programa -t                    # Tutorial completo\n");
}

void print_color(const char* color, const char* format, ...) {
    va_list args;
    printf("%s", color);
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    printf("%s", COLOR_RESET);
}

void print_bullet(const char* text, int indent) {
    for (int i = 0; i < indent; i++) printf("  ");
    printf("• %s\n", text);
}

void print_section(const char* title) {
    printf("\n");
    print_color(COLOR_CYAN, "▸ %s\n", title);
    for (int i = 0; i < strlen(title) + 2; i++) printf("─");
    printf("\n");
}

void print_note(const char* note) {
    print_color(COLOR_BLUE, "📝 Nota: %s\n", note);
}

void print_warning(const char* warning) {
    print_color(COLOR_YELLOW, "⚠️  Advertencia: %s\n", warning);
}

void print_success(const char* message) {
    print_color(COLOR_GREEN, "✅ %s\n", message);
}

void print_error(const char* error) {
    print_color(COLOR_RED, "❌ Error: %s\n", error);
}

void center_text(const char* text) {
    int padding = (terminal_width - strlen(text)) / 2;
    if (padding < 0) padding = 0;
    for (int i = 0; i < padding; i++) printf(" ");
    printf("%s\n", text);
}

void print_welcome_message() {
    clear_screen();
    print_separator('=');
    center_text("🤖 SISTEMA SVM DIDACTICO - APRENDE SUPPORT VECTOR MACHINES");
    print_separator('=');
    
    printf("\n");
    center_text("Una herramienta educativa completa para entender SVM desde cero");
    printf("\n");
    
    print_color(COLOR_MAGENTA, "🎯 Caracteristicas principales:\n");
    print_bullet("Entrenamiento visual paso a paso", 1);
    print_bullet("Explicaciones detalladas de cada concepto", 1);
    print_bullet("Modo aprendizaje con preguntas y respuestas", 1);
    print_bullet("Analisis de errores y diagnostico de modelos", 1);
    print_bullet("Persistencia de modelos y datasets", 1);
    print_bullet("Generacion de informes completos", 1);
    
    printf("\n");
    print_color(COLOR_YELLOW, "💡 Consejo: Usa el modo -learn para una experiencia educativa guiada.\n");
    printf("\n");
}

void setup_learning_environment() {
    print_header("🎓 MODO APRENDIZAJE ACTIVO");
    
    printf("\nBienvenido al modo aprendizaje activo. En este modo:\n");
    print_bullet("Cada concepto se explica detalladamente", 1);
    print_bullet("Podras ver el entrenamiento paso a paso", 1);
    print_bullet("Responderas preguntas para reforzar tu comprension", 1);
    print_bullet("Analizaras errores comunes", 1);
    print_bullet("Obtendras recomendaciones personalizadas", 1);
    
    printf("\n");
    print_color(COLOR_GREEN, "¿Estas listo para comenzar tu aprendizaje? (s/n): ");
    
    char respuesta;
    scanf("%c", &respuesta);
    getchar();
    
    if (respuesta == 's' || respuesta == 'S') {
        learning_mode = 2;  // Modo paso a paso completo
        print_success("¡Excelente! Comenzando experiencia de aprendizaje...\n");
        sleep(2);
    } else {
        learning_mode = 1;  // Solo explicaciones
        print_note("Modo aprendizaje con solo explicaciones habilitado.\n");
        sleep(1);
    }
}

void print_progress_bar(int current, int total, const char* label, double value, int color) {
    int bar_width = 40;
    int pos = (int)((double)current / total * bar_width);
    
    printf("\r%s [", label);
    for (int i = 0; i < bar_width; i++) {
        if (i < pos) printf("#");
        else if (i == pos) printf(">");
        else printf(" ");
    }
    printf("] %d/%d", current, total);
    
    if (value >= 0) {
        if (color == 1) printf(" | ");
        else printf(" | ");
        printf("Valor: %.4f", value);
    }
    
    fflush(stdout);
    
    if (current == total) {
        printf("\n");
    }
}

void print_svm_visualization(Dataset* dataset, SVM_Model* model) {
    if (dataset->num_features < 2) {
        print_error("Se necesitan al menos 2 caracteristicas para visualizar");
        return;
    }
    
    print_section("VISUALIZACION DEL MODELO SVM");
    
    printf("Leyenda:\n");
    print_color(COLOR_BLUE, "  + = Clase +1 (positiva)\n");
    print_color(COLOR_RED, "  - = Clase -1 (negativa)\n");
    print_color(COLOR_YELLOW, "  * = Vector soporte\n");
    print_color(COLOR_GREEN, "  | = Hiperplano de decision\n");
    print_color(COLOR_WHITE, "  . = Margen del hiperplano\n");
    
    printf("\n");
    
    int grid_size = 40;
    char grid[grid_size][grid_size];
    
    // Inicializar grid
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            grid[i][j] = ' ';
        }
    }
    
    // Calcular limites
    double min_x = dataset->feature_min[0];
    double max_x = dataset->feature_max[0];
    double min_y = dataset->feature_min[1];
    double max_y = dataset->feature_max[1];
    
    // Dibujar regiones de decision
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            double x = min_x + (max_x - min_x) * j / (grid_size - 1);
            double y = min_y + (max_y - min_y) * (grid_size - 1 - i) / (grid_size - 1);
            
            DataPoint test_point = {0};
            test_point.features[0] = x;
            test_point.features[1] = y;
            
            double prediction = predict_point(model, &test_point);
            
            // Dibujar regiones
            if (prediction > 1.0) {
                grid[i][j] = '+';  // Region positiva fuerte
            } else if (prediction > 0) {
                grid[i][j] = 'o';  // Region positiva debil
            } else if (prediction < -1.0) {
                grid[i][j] = '-';  // Region negativa fuerte
            } else if (prediction < 0) {
                grid[i][j] = 'x';  // Region negativa debil
            }
            
            // Dibujar hiperplano y margen
            if (fabs(prediction) < 0.05) {
                grid[i][j] = '|';  // Hiperplano
            } else if (fabs(prediction - 1.0) < 0.1 || fabs(prediction + 1.0) < 0.1) {
                grid[i][j] = '.';  // Margen
            }
        }
    }
    
    // Dibujar puntos de datos reales
    for (int s = 0; s < dataset->num_samples && s < 100; s++) {
        int x = (int)((dataset->points[s].features[0] - min_x) / (max_x - min_x) * (grid_size - 1));
        int y = (int)((dataset->points[s].features[1] - min_y) / (max_y - min_y) * (grid_size - 1));
        y = grid_size - 1 - y;
        
        if (x >= 0 && x < grid_size && y >= 0 && y < grid_size) {
            char symbol = (dataset->points[s].label == 1) ? '+' : '-';
            
            // Marcar vectores soporte
            for (int v = 0; v < model->num_support_vectors; v++) {
                if (&dataset->points[s] == &model->support_vectors[v]) {
                    symbol = '*';
                    dataset->points[s].is_support_vector = 1;
                    break;
                }
            }
            
            // Marcar errores de clasificacion
            double prediction = predict_point(model, &dataset->points[s]);
            int predicted_label = (prediction >= 0) ? 1 : -1;
            if (predicted_label != dataset->points[s].label) {
                symbol = (symbol == '+') ? 'P' : 'N';  // P para falso positivo, N para falso negativo
                dataset->points[s].is_misclassified = 1;
            }
            
            grid[y][x] = symbol;
        }
    }
    
    // Imprimir grid
    printf("   y\n");
    for (int i = 0; i < grid_size; i++) {
        printf("%3d ", grid_size - i - 1);
        for (int j = 0; j < grid_size; j++) {
            char c = grid[i][j];
            switch(c) {
                case '+': printf("\033[34m%c\033[0m", c); break;
                case '-': printf("\033[31m%c\033[0m", c); break;
                case '*': printf("\033[33m%c\033[0m", c); break;
                case '|': printf("\033[32m%c\033[0m", c); break;
                case '.': printf("\033[90m%c\033[0m", c); break;
                case 'o': printf("\033[36m%c\033[0m", c); break;
                case 'x': printf("\033[35m%c\033[0m", c); break;
                case 'P': printf("\033[41mP\033[0m"); break;  // Falso positivo
                case 'N': printf("\033[41mN\033[0m"); break;  // Falso negativo
                default: printf("%c", c);
            }
        }
        printf("\n");
    }
    
    // Eje X
    printf("    ");
    for (int j = 0; j < grid_size; j += 5) {
        printf("+----");
    }
    printf("\n    ");
    for (int j = 0; j < grid_size; j += 10) {
        printf("%-8.1f", min_x + (max_x - min_x) * j / (grid_size - 1));
    }
    printf(" x\n");
    
    // Estadisticas de visualizacion
    printf("\nEstadisticas en visualizacion:\n");
    printf("  • Puntos mostrados: %d/%d\n", 
           dataset->num_samples < 100 ? dataset->num_samples : 100,
           dataset->num_samples);
    printf("  • Resolucion: %dx%d\n", grid_size, grid_size);
    printf("  • Rango X: [%.2f, %.2f]\n", min_x, max_x);
    printf("  • Rango Y: [%.2f, %.2f]\n", min_y, max_y);
}

void print_support_vectors_visualization(SVM_Model* model) {
    print_section("VECTORES SOPORTE DETALLADOS");
    
    printf("Cantidad total: %d vectores soporte\n\n", model->num_support_vectors);
    
    printf("┌─────┬──────────┬────────────┬────────────┬──────────────┐\n");
    printf("│ No. │ Etiqueta │   Alpha    │ Distancia  │ Influencia   │\n");
    printf("├─────┼──────────┼────────────┼────────────┼──────────────┤\n");
    
    double total_influence = 0.0;
    for (int i = 0; i < model->num_support_vectors && i < 15; i++) {
        double distance = compute_distance_to_hyperplane(model, &model->support_vectors[i]);
        double influence = fabs(model->support_vectors[i].alpha * distance);
        total_influence += influence;
        
        printf("│ %3d │    %+2d    │ %10.4f │ %10.4f │ %12.4f │\n",
               i + 1,
               model->support_vectors[i].label,
               model->support_vectors[i].alpha,
               distance,
               influence);
    }
    
    printf("└─────┴──────────┴────────────┴────────────┴──────────────┘\n");
    
    // Estadisticas
    if (model->num_support_vectors > 0) {
        printf("\n📊 Estadisticas de vectores soporte:\n");
        
        double avg_alpha = 0.0, max_alpha = -DBL_MAX, min_alpha = DBL_MAX;
        int pos_sv = 0, neg_sv = 0;
        
        for (int i = 0; i < model->num_support_vectors; i++) {
            avg_alpha += fabs(model->support_vectors[i].alpha);
            if (model->support_vectors[i].alpha > max_alpha) max_alpha = model->support_vectors[i].alpha;
            if (model->support_vectors[i].alpha < min_alpha) min_alpha = model->support_vectors[i].alpha;
            if (model->support_vectors[i].label == 1) pos_sv++; else neg_sv++;
        }
        avg_alpha /= model->num_support_vectors;
        
        printf("  • Alpha promedio: %.4f\n", avg_alpha);
        printf("  • Alpha maximo:   %.4f\n", max_alpha);
        printf("  • Alpha minimo:   %.4f\n", min_alpha);
        printf("  • Vectores +1:    %d (%.1f%%)\n", pos_sv, (double)pos_sv/model->num_support_vectors*100);
        printf("  • Vectores -1:    %d (%.1f%%)\n", neg_sv, (double)neg_sv/model->num_support_vectors*100);
        printf("  • Influencia total: %.4f\n", total_influence);
        printf("  • Porcentaje del dataset: %.1f%%\n", 
               (double)model->num_support_vectors/current_dataset.num_samples*100);
    }
    
    // Explicacion educativa
    if (learning_mode >= 1) {
        printf("\n💡 Explicacion:\n");
        printf("Los vectores soporte son los puntos mas cercanos al hiperplano.\n");
        printf("Alpha (α) es el multiplicador de Lagrange que indica la importancia\n");
        printf("de cada vector soporte. Valores mas altos indican mayor influencia.\n");
    }
}

void print_training_progress(SVM_Model* model, int iteration, double error, double accuracy) {
    static clock_t start_time = 0;
    if (iteration == 0) start_time = clock();
    
    if (iteration % 50 == 0 || iteration == model->iterations - 1) {
        double elapsed = ((double)(clock() - start_time)) / CLOCKS_PER_SEC;
        
        clear_screen();
        print_header("ENTRENAMIENTO EN PROGRESO");
        
        printf("\nIteracion: %d/%d\n", iteration + 1, model->iterations);
        printf("Tiempo transcurrido: %.2f segundos\n", elapsed);
        printf("Error promedio: %.6f\n", error);
        printf("Vectores soporte: %d\n", model->num_support_vectors);
        printf("Margen: %.4f\n", model->margin);
        printf("Exactitud: %.2f%%\n", accuracy * 100);
        
        // Grafico de evolucion
        printf("\n📈 Evolucion del entrenamiento:\n");
        
        int graph_height = 12;
        int graph_width = 60;
        
        // Normalizar valores para grafico
        double max_error = 0.0;
        double max_acc = 0.0;
        for (int i = 0; i <= iteration && i < 100; i++) {
            if (training_history.error_history[i] > max_error) max_error = training_history.error_history[i];
            if (training_history.accuracy_history[i] > max_acc) max_acc = training_history.accuracy_history[i];
        }
        if (max_error < 0.001) max_error = 1.0;
        if (max_acc < 0.001) max_acc = 1.0;
        
        // Grafico combinado
        for (int h = graph_height; h >= 0; h--) {
            printf("%3.0f%% │", (double)h / graph_height * 100);
            
            for (int i = 0; i <= iteration && i < graph_width; i++) {
                double error_norm = training_history.error_history[i] / max_error;
                double acc_norm = training_history.accuracy_history[i] / max_acc;
                double sv_norm = (double)training_history.sv_count_history[i] / current_dataset.num_samples;
                
                int error_pos = (int)(error_norm * graph_height);
                int acc_pos = (int)(acc_norm * graph_height);
                int sv_pos = (int)(sv_norm * graph_height);
                
                char symbol = ' ';
                if (h == error_pos && h == acc_pos && h == sv_pos) {
                    symbol = '@';  // Cambiado de '★' a '@' para evitar warning
                } else if (h == error_pos && h == acc_pos) {
                    symbol = 'X';
                } else if (h == error_pos && h == sv_pos) {
                    symbol = 'E';
                } else if (h == acc_pos && h == sv_pos) {
                    symbol = 'A';
                } else if (h == error_pos) {
                    symbol = 'E';
                } else if (h == acc_pos) {
                    symbol = 'A';
                } else if (h == sv_pos) {
                    symbol = 'S';
                } else if (h == 0) {
                    symbol = '_';
                }
                
                // Colorear segun simbolo
                switch(symbol) {
                    case 'E': printf("\033[31mE\033[0m"); break;
                    case 'A': printf("\033[32mA\033[0m"); break;
                    case 'S': printf("\033[33mS\033[0m"); break;
                    case 'X': printf("\033[35mX\033[0m"); break;
                    case '@': printf("\033[36m@\033[0m"); break;
                    case '_': printf("_"); break;
                    default: printf(" ");
                }
            }
            printf("\n");
        }
        
        printf("     ");
        for (int i = 0; i < 6; i++) {
            printf("%-10d", (i+1) * (graph_width/6));
        }
        printf(" Iteraciones\n");
        printf("     Leyenda: E=Error, A=Exactitud, S=Vectores Soporte\n");
        
        // Tiempo estimado restante
        if (iteration > 10 && iteration < model->iterations - 1) {
            double estimated_total = elapsed * model->iterations / (iteration + 1);
            double remaining = estimated_total - elapsed;
            printf("\n⏱️  Tiempo estimado restante: %.1f segundos\n", remaining);
        }
        
        // Pausa para efecto visual (solo en modo aprendizaje)
        if (learning_mode == 2 && iteration % 100 == 0 && iteration < model->iterations - 100) {
            printf("\n");
            wait_for_key("Pausa para observacion...");
        } else {
            usleep(50000);  // 50ms pausa
        }
    }
}

void print_hyperplane_equation(SVM_Model* model, Dataset* dataset) {
    print_section("ECUACION DEL MODELO SVM");
    
    if (strcmp(model->kernel_type, "linear") == 0) {
        printf("Para kernel LINEAL:\n\n");
        printf("   f(x) = ");
        
        int terms_printed = 0;
        for (int i = 0; i < dataset->num_features && i < 6; i++) {
            if (fabs(model->weights[i]) > 0.0001) {
                if (terms_printed > 0 && model->weights[i] >= 0) printf(" + ");
                else if (model->weights[i] < 0) printf(" - ");
                
                printf("%.4f·%s", fabs(model->weights[i]), dataset->feature_names[i]);
                terms_printed++;
            }
        }
        
        if (fabs(model->bias) > 0.0001) {
            if (model->bias >= 0) printf(" + %.4f", model->bias);
            else printf(" - %.4f", -model->bias);
        }
        
        printf("\n\n");
        printf("Donde:\n");
        printf("  • f(x) >  0 → Clase +1\n");
        printf("  • f(x) =  0 → Hiperplano de decision\n");
        printf("  • f(x) = ±1 → Limites del margen\n");
        printf("  • f(x) <  0 → Clase -1\n");
        
        printf("\nInterpretacion geometrica:\n");
        printf("  • ||w|| = %.4f (norma del vector peso)\n", sqrt(dot_product(model->weights, model->weights, dataset->num_features)));
        printf("  • Margen = 1/||w|| = %.4f\n", model->margin);
        printf("  • w apunta hacia la clase positiva\n");
        
    } else if (strcmp(model->kernel_type, "rbf") == 0) {
        printf("Para kernel RBF (Radial Basis Function):\n\n");
        printf("   f(x) = Σ α_i·y_i·exp(-γ·||x - x_i||²) + b\n");
        printf("   donde:\n");
        printf("     • γ = %.4f (parametro del kernel)\n", model->gamma);
        printf("     • x_i son los vectores soporte\n");
        printf("     • α_i son los multiplicadores de Lagrange\n");
        printf("     • y_i ∈ {-1, +1} son las etiquetas\n");
        printf("     • b = %.4f (sesgo)\n", model->bias);
        
        printf("\nCaracteristicas del kernel RBF:\n");
        printf("  • Creae con la similitud entre puntos\n");
        printf("  • γ controla el radio de influencia\n");
        printf("  • Puede crear fronteras no lineales complejas\n");
        
    } else if (strcmp(model->kernel_type, "poly") == 0) {
        printf("Para kernel POLINOMIAL:\n\n");
        printf("   K(x, y) = (x·y + 1)^%.0f\n", model->degree);
        printf("   f(x) = Σ α_i·y_i·K(x, x_i) + b\n");
        
        printf("\nGrado del polinomio: %.0f\n", model->degree);
        printf("Puede modelar relaciones polinomiales de grado %d\n", (int)model->degree);
    }
    
    // Explicacion educativa
    if (learning_mode >= 1) {
        printf("\n🎓 Explicacion para aprendizaje:\n");
        printf("La ecuacion del hiperplano define como el SVM toma decisiones.\n");
        printf("Cada caracteristica contribuye segun su peso (w_i).\n");
        printf("El sesgo (b) desplaza el hiperplano del origen.\n");
        printf("El signo de f(x) determina la clase predicha.\n");
    }
}

void print_margin_visualization(Dataset* dataset, SVM_Model* model) {
    print_section("ANALISIS DEL MARGEN");
    
    printf("Concepto de margen en SVM:\n");
    printf("El margen es la distancia entre el hiperplano y los vectores soporte mas cercanos.\n");
    printf("SVM busca MAXIMIZAR este margen para mejorar la generalizacion.\n\n");
    
    // Visualizacion ASCII del margen
    printf("         Clase -1                 Clase +1\n");
    printf("   ┌─────────────────┐      ┌─────────────────┐\n");
    printf("   │                 │      │                 │\n");
    printf("   │   ●   ●   ●     │      │   +   +   +     │\n");
    printf("   │     ●     ●     │      │     +     +     │\n");
    printf("   │       ●   ●     │      │       +   +     │\n");
    printf("   │         ●       │      │         +       │\n");
    printf("   └─────────┼───────┘      └───────┼─────────┘\n");
    printf("             │      MARGEN = %.3f     │\n", model->margin);
    printf("             ├───────────────────────┤\n");
    printf("             │     HIPERPLANO        │\n");
    printf("             │      f(x) = 0         │\n");
    printf("             ├───────────────────────┤\n");
    printf("             │                       │\n");
    printf("   ★ = Vector soporte de clase -1\n");
    printf("   ☆ = Vector soporte de clase +1\n");
    printf("   ● = Puntos de clase -1\n");
    printf("   + = Puntos de clase +1\n");
    
    // Calculos detallados del margen
    printf("\n📏 Calculos del margen:\n");
    
    if (strcmp(model->kernel_type, "linear") == 0) {
        double norm_w = 0.0;
        for (int i = 0; i < model->num_features_trained; i++) {
            norm_w += model->weights[i] * model->weights[i];
        }
        norm_w = sqrt(norm_w);
        
        printf("  • ||w|| = √(");
        for (int i = 0; i < model->num_features_trained && i < 3; i++) {
            if (i > 0) printf(" + ");
            printf("%.2f²", model->weights[i]);
        }
        if (model->num_features_trained > 3) printf(" + ...");
        printf(") = %.4f\n", norm_w);
        
        printf("  • Margen teorico = 1 / ||w|| = 1 / %.4f = %.4f\n", norm_w, 1.0/norm_w);
        printf("  • Margen calculado = %.4f\n", model->margin);
        
        if (fabs(model->margin - 1.0/norm_w) > 0.001) {
            printf("  • Diferencia: %.4f (puede deberse a vectores soporte no exactos)\n", 
                   fabs(model->margin - 1.0/norm_w));
        }
    }
    
    // Analisis de vectores soporte y margen
    printf("\n🔍 Analisis de los vectores soporte:\n");
    
    if (model->num_support_vectors > 0) {
        double min_distance = DBL_MAX;
        double max_distance = 0.0;
        DataPoint* closest_sv = NULL;
        
        for (int i = 0; i < model->num_support_vectors; i++) {
            double distance = compute_distance_to_hyperplane(model, &model->support_vectors[i]);
            if (fabs(distance) < min_distance) {
                min_distance = fabs(distance);
                closest_sv = &model->support_vectors[i];
            }
            if (fabs(distance) > max_distance) max_distance = fabs(distance);
        }
        
        printf("  • Distancia minima al hiperplano: %.4f\n", min_distance);
        printf("  • Distancia maxima al hiperplano: %.4f\n", max_distance);
        printf("  • Vector soporte mas cercano: ");
        if (closest_sv) {
            printf("[");
            for (int i = 0; i < model->num_features_trained && i < 2; i++) {
                if (i > 0) printf(", ");
                printf("%.2f", closest_sv->features[i]);
            }
            printf("] con α=%.4f\n", closest_sv->alpha);
        }
        
        // Verificar si el margen es "duro" o "blando"
        int on_margin = 0;
        for (int i = 0; i < model->num_support_vectors; i++) {
            double distance = fabs(compute_distance_to_hyperplane(model, &model->support_vectors[i]));
            if (fabs(distance - 1.0) < 0.1) on_margin++;
        }
        
        double percentage_on_margin = (double)on_margin / model->num_support_vectors * 100;
        printf("  • Vectores en el margen: %d/%d (%.1f%%)\n", on_margin, model->num_support_vectors, percentage_on_margin);
        
        if (percentage_on_margin > 80) {
            printf("  • Tipo: Margen DURO (la mayoria esta justo en el margen)\n");
        } else {
            printf("  • Tipo: Margen BLANDO (algunos estan dentro del margen)\n");
        }
    }
    
    // Efecto del parametro C en el margen
    printf("\n⚙️  Efecto del parametro C (C = %.2f):\n", model->C);
    if (model->C < 0.1) {
        printf("  • C muy bajo → Margen grande, pocos errores permitidos\n");
        printf("  • Prioriza maximizar el margen sobre clasificar todo correctamente\n");
    } else if (model->C > 10) {
        printf("  • C muy alto → Margen pequeno, muchos errores permitidos\n");
        printf("  • Prioriza clasificar correctamente sobre tener margen grande\n");
    } else {
        printf("  • C moderado → Balance entre margen y errores\n");
        printf("  • Compromiso optimo para generalizacion\n");
    }
    
    // Recomendaciones
    printf("\n💡 Recomendaciones:\n");
    if (model->margin < 0.1) {
        printf("  • El margen es muy pequeno, considera aumentar C\n");
        printf("  • Podria haber sobreajuste a los datos de entrenamiento\n");
    } else if (model->margin > 1.0) {
        printf("  • El margen es muy grande, considera disminuir C\n");
        printf("  • El modelo podria ser demasiado simple\n");
    } else {
        printf("  • Margen en rango adecuado\n");
        printf("  • El modelo deberia generalizar bien\n");
    }
}

void print_kernel_comparison() {
    print_section("COMPARACION COMPLETA DE KERNELS");
    
    printf("Los kernels transforman los datos a espacios de mayor dimension\n");
    printf("donde se vuelven linealmente separables (\"kernel trick\").\n\n");
    
    printf("┌──────────────┬──────────────────────────────┬──────────────────────────┬──────────────────────┐\n");
    printf("│   Kernel     │         Formula              │      Ventajas            │    Desventajas       │\n");
    printf("├──────────────┼──────────────────────────────┼──────────────────────────┼──────────────────────┤\n");
    printf("│   Lineal     │ K(x,y) = x·y                │ • Simple y rapido        │ • Solo separa       │\n");
    printf("│              │                              │ • Pocos parametros       │   linealmente        │\n");
    printf("│              │                              │ • Interpretable          │ • Limitado          │\n");
    printf("├──────────────┼──────────────────────────────┼──────────────────────────┼──────────────────────┤\n");
    printf("│   RBF        │ K(x,y)=exp(-γ‖x-y‖²)        │ • Separa cualquier      │ • Computacionalmente│\n");
    printf("│   (Gaussiano)│                              │   conjunto               │   costoso           │\n");
    printf("│              │                              │ • Un solo parametro (γ)  │ • Sensible a γ      │\n");
    printf("│              │                              │ • Buen default           │ • Menos interpretable│\n");
    printf("├──────────────┼──────────────────────────────┼──────────────────────────┼──────────────────────┤\n");
    printf("│ Polinomial   │ K(x,y)=(x·y+c)^d            │ • Captura relaciones     │ • Mas parametros    │\n");
    printf("│              │                              │   polinomiales           │ • Inestable con     │\n");
    printf("│              │                              │ • Flexible con grado d   │   alto grado        │\n");
    printf("└──────────────┴──────────────────────────────┴──────────────────────────┴──────────────────────┘\n");
    
    printf("\n🎯 GUIA DE SELECCION DE KERNEL:\n");
    printf("\n1. EMPIEZA CON LINEAL SI:\n");
    printf("   • Tienes muchas caracteristicas (>1000)\n");
    printf("   • Los datos son linealmente separables\n");
    printf("   • Necesitas un modelo rapido y simple\n");
    printf("   • La interpretabilidad es importante\n");
    
    printf("\n2. USA RBF (GAUSSIANO) SI:\n");
    printf("   • No sabes que kernel usar (default)\n");
    printf("   • Los datos no son linealmente separables\n");
    printf("   • Tienes un conjunto de datos pequeno-mediano\n");
    printf("   • Quieres buenos resultados sin mucho ajuste\n");
    
    printf("\n3. USA POLINOMIAL SI:\n");
    printf("   • Sabes que la relacion es polinomial\n");
    printf("   • Tienes conocimiento del dominio\n");
    printf("   • Quieres control explicito del grado\n");
    printf("   • Estas dispuesto a ajustar mas parametros\n");
    
    printf("\n⚙️  AJUSTE DE HIPERPARAMETROS:\n");
    printf("LINEAL: Solo ajusta C (trade-off margen/error)\n");
    printf("RBF: Ajusta C y γ (gamma)\n");
    printf("     • γ alto → Fronteras complejas (riesgo overfit)\n");
    printf("     • γ bajo → Fronteras suaves (riesgo underfit)\n");
    printf("POLINOMIAL: Ajusta C, grado (d) y coeficiente (c)\n");
    
    printf("\n🔍 METODOLOGIA RECOMENDADA:\n");
    printf("1. Dividir datos en entrenamiento/validacion/prueba\n");
    printf("2. Empezar con kernel RBF y valores por defecto\n");
    printf("3. Usar validacion cruzada para ajustar C y γ\n");
    printf("4. Probar kernel lineal como baseline\n");
    printf("5. Elegir el modelo con mejor validacion\n");
    printf("6. Evaluar final con conjunto de prueba\n");
    
    if (learning_mode >= 1) {
        printf("\n🎓 EJERCICIO PRACTICO:\n");
        printf("Intenta entrenar modelos con los tres kernels en el mismo dataset\n");
        printf("y compara sus resultados. Observa como cambian las fronteras de\n");
        printf("decision y el numero de vectores soporte en cada caso.\n");
    }
}

void print_model_info(SVM_Model* model) {
    print_section("INFORMACION COMPLETA DEL MODELO");
    
    printf("📋 INFORMACION BASICA:\n");
    printf("  • Nombre: %s\n", model->name);
    printf("  • Kernel: %s\n", model->kernel_type);
    printf("  • Fecha de entrenamiento: %s", ctime(&model->trained_at));
    printf("  • Tiempo de entrenamiento: %.2f segundos\n", model->training_time);
    printf("  • Iteraciones: %d\n", model->iterations);
    printf("  • Caracteristicas usadas: %d\n", model->num_features_trained);
    
    printf("\n📊 METRICAS DE RENDIMIENTO:\n");
    printf("  • Exactitud entrenamiento: %.2f%%\n", model->training_accuracy * 100);
    printf("  • Exactitud validacion: %.2f%%\n", model->validation_accuracy * 100);
    printf("  • Exactitud general: %.2f%%\n", model->accuracy * 100);
    printf("  • Margen: %.4f\n", model->margin);
    printf("  • Vectores soporte: %d (%.1f%% del dataset)\n", 
           model->num_support_vectors,
           (double)model->num_support_vectors/current_dataset.num_samples*100);
    
    printf("\n⚙️  PARAMETROS DEL MODELO:\n");
    printf("  • C (regularizacion): %.4f\n", model->C);
    if (strcmp(model->kernel_type, "rbf") == 0) {
        printf("  • γ (gamma - RBF): %.4f\n", model->gamma);
    } else if (strcmp(model->kernel_type, "poly") == 0) {
        printf("  • Grado (polinomial): %.0f\n", model->degree);
    }
    printf("  • Bias (sesgo): %.4f\n", model->bias);
    
    printf("\n📈 COMPLEJIDAD DEL MODELO:\n");
    if (strcmp(model->kernel_type, "linear") == 0) {
        printf("  • Tipo: Modelo lineal simple\n");
        printf("  • Parametros libres: %d pesos + 1 bias = %d\n", 
               model->num_features_trained, model->num_features_trained + 1);
        printf("  • Capacidad: Baja (solo fronteras lineales)\n");
    } else if (strcmp(model->kernel_type, "rbf") == 0) {
        printf("  • Tipo: Modelo no lineal (RBF)\n");
        printf("  • Parametros: %d vectores soporte * α_i\n", model->num_support_vectors);
        printf("  • Capacidad: Alta (fronteras complejas)\n");
        printf("  • Radio efectivo: 1/√γ = %.4f\n", 1.0/sqrt(model->gamma));
    }
    
    printf("\n🔍 DIAGNOSTICO RAPIDO:\n");
    
    // Verificar sobreajuste
    double overfit_margin = model->training_accuracy - model->validation_accuracy;
    if (overfit_margin > 0.15) {
        printf("  • ⚠️  POSIBLE SOBREAJUSTE: Diferencia entrenamiento-validacion > 15%%\n");
        printf("    Considera: Aumentar C, reducir γ (RBF), o conseguir mas datos\n");
    } else if (overfit_margin > 0.05) {
        printf("  • ⚠️  Leve sobreajuste\n");
    } else {
        printf("  • ✅ Buen balance entrenamiento-validacion\n");
    }
    
    // Verificar margen
    if (model->margin < 0.05) {
        printf("  • ⚠️  Margen muy pequeno (riesgo de sobreajuste)\n");
    } else if (model->margin > 2.0) {
        printf("  • ⚠️  Margen muy grande (riesgo de subajuste)\n");
    } else {
        printf("  • ✅ Margen en rango adecuado\n");
    }
    
    // Verificar vectores soporte
    double sv_percentage = (double)model->num_support_vectors/current_dataset.num_samples;
    if (sv_percentage > 0.5) {
        printf("  • ⚠️  Muchos vectores soporte (>50%% del dataset)\n");
        printf("    El modelo podria estar memorizando los datos\n");
    } else if (sv_percentage < 0.05) {
        printf("  • ⚠️  Pocos vectores soporte (<5%%)\n");
        printf("    El modelo podria ser demasiado simple\n");
    } else {
        printf("  • ✅ Porcentaje de vectores soporte adecuado\n");
    }
    
    printf("\n💡 RECOMENDACIONES:\n");
    if (strcmp(model->kernel_type, "linear") == 0 && model->accuracy < 0.7) {
        printf("  • Prueba kernel RBF para datos no lineales\n");
    }
    if (model->C < 0.1) {
        printf("  • Considera aumentar C para permitir mas errores\n");
    }
    if (model->num_support_vectors == current_dataset.num_samples) {
        printf("  • Todos los puntos son vectores soporte\n");
        printf("  • Considera aumentar C o cambiar kernel\n");
    }
}

void print_dataset_visualization(Dataset* dataset) {
    if (dataset->num_samples == 0) {
        print_error("Dataset vacio");
        return;
    }
    
    print_section("VISUALIZACION DEL DATASET");
    
    printf("Informacion general:\n");
    printf("  • Muestras: %d\n", dataset->num_samples);
    printf("  • Caracteristicas: %d\n", dataset->num_features);
    printf("  • Nombre: %s\n", dataset->name);
    
    if (strlen(dataset->description) > 0) {
        printf("  • Descripcion: %s\n", dataset->description);
    }
    
    // Distribucion de clases
    int class_pos = 0, class_neg = 0;
    for (int i = 0; i < dataset->num_samples; i++) {
        if (dataset->points[i].label == 1) class_pos++;
        else class_neg++;
    }
    
    printf("\n📊 DISTRIBUCION DE CLASES:\n");
    printf("  • Clase +1: %d muestras (%.1f%%)\n", class_pos, (double)class_pos/dataset->num_samples*100);
    printf("  • Clase -1: %d muestras (%.1f%%)\n", class_neg, (double)class_neg/dataset->num_samples*100);
    
    // Visualizacion simple si hay 2 caracteristicas
    if (dataset->num_features >= 2) {
        printf("\n📈 DISTRIBUCION EN 2D (primeras 2 caracteristicas):\n");
        
        int grid_size = 50;
        char grid[grid_size][grid_size];
        for (int i = 0; i < grid_size; i++) {
            for (int j = 0; j < grid_size; j++) {
                grid[i][j] = ' ';
            }
        }
        
        double min_x = dataset->feature_min[0];
        double max_x = dataset->feature_max[0];
        double min_y = dataset->feature_min[1];
        double max_y = dataset->feature_max[1];
        
        // Contar puntos por celda - CORRECCION: usar memset para inicializar
        int counts[grid_size][grid_size][2];
        memset(counts, 0, sizeof(counts));
        
        for (int s = 0; s < dataset->num_samples; s++) {
            int x = (int)((dataset->points[s].features[0] - min_x) / (max_x - min_x) * (grid_size - 1));
            int y = (int)((dataset->points[s].features[1] - min_y) / (max_y - min_y) * (grid_size - 1));
            y = grid_size - 1 - y;
            
            if (x >= 0 && x < grid_size && y >= 0 && y < grid_size) {
                if (dataset->points[s].label == 1) counts[y][x][0]++;
                else counts[y][x][1]++;
            }
        }
        
        // Crear visualizacion
        for (int i = 0; i < grid_size; i++) {
            for (int j = 0; j < grid_size; j++) {
                int pos = counts[i][j][0];
                int neg = counts[i][j][1];
                
                if (pos > 0 && neg > 0) {
                    grid[i][j] = 'X';  // Mezcla
                } else if (pos > 0) {
                    grid[i][j] = '+';
                } else if (neg > 0) {
                    grid[i][j] = '-';
                } else if ((i % 5 == 0 && j % 5 == 0) || (i == 0 || j == 0 || i == grid_size-1 || j == grid_size-1)) {
                    grid[i][j] = '.';
                }
            }
        }
        
        // Imprimir grid
        printf("    y\n");
        for (int i = 0; i < grid_size; i++) {
            printf("%3d ", grid_size - i - 1);
            for (int j = 0; j < grid_size; j++) {
                char c = grid[i][j];
                if (c == '+') printf("\033[34m%c\033[0m", c);
                else if (c == '-') printf("\033[31m%c\033[0m", c);
                else if (c == 'X') printf("\033[33m%c\033[0m", c);
                else if (c == '.') printf("\033[90m%c\033[0m", c);
                else printf("%c", c);
            }
            printf("\n");
        }
        
        printf("    ");
        for (int j = 0; j < grid_size; j += 5) {
            printf("+----");
        }
        printf("\n    ");
        for (int j = 0; j < grid_size; j += 10) {
            printf("%-8.1f", min_x + (max_x - min_x) * j / (grid_size - 1));
        }
        printf(" %s\n", dataset->feature_names[0]);
        
        printf("\nLeyenda:\n");
        printf("  • %s = Solo clase +1\n", COLOR_BLUE);
        printf("  • %s = Solo clase -1\n", COLOR_RED);
        printf("  • %s = Ambas clases mezcladas\n", COLOR_YELLOW);
        printf("  • %s = Puntos de referencia\n", COLOR_RESET);
    }
    
    // Estadisticas por caracteristica
    printf("\n📐 ESTADISTICAS POR CARACTERISTICA:\n");
    printf("┌─────┬──────────────────────┬────────────┬────────────┬────────────┐\n");
    printf("│ No. │ Nombre               │   Minimo   │   Maximo   │   Media    │\n");
    printf("├─────┼──────────────────────┼────────────┼────────────┼────────────┤\n");
    
    for (int i = 0; i < dataset->num_features && i < 8; i++) {
        double sum = 0.0;
        for (int j = 0; j < dataset->num_samples; j++) {
            sum += dataset->points[j].features[i];
        }
        double mean = sum / dataset->num_samples;
        
        printf("│ %3d │ %-20s │ %10.4f │ %10.4f │ %10.4f │\n",
               i + 1,
               dataset->feature_names[i],
               dataset->feature_min[i],
               dataset->feature_max[i],
               mean);
    }
    
    printf("└─────┴──────────────────────┴────────────┴────────────┴────────────┘\n");
    
    // Analisis de separabilidad
    if (dataset->num_features >= 2) {
        printf("\n🔍 ANALISIS DE SEPARABILIDAD LINEAL (estimado):\n");
        
        // Estimacion simple de separabilidad lineal
        int linear_errors = 0;
        for (int i = 0; i < dataset->num_samples; i++) {
            for (int j = i + 1; j < dataset->num_samples; j++) {
                // Verificar si puntos de diferente clase estan mezclados
                if (dataset->points[i].label != dataset->points[j].label) {
                    double dist = 0.0;
                    for (int k = 0; k < dataset->num_features; k++) {
                        double diff = dataset->points[i].features[k] - dataset->points[j].features[k];
                        dist += diff * diff;
                    }
                    if (dist < 0.1) linear_errors++;  // Puntos muy cercanos de diferente clase
                }
            }
        }
        
        double separability_score = 1.0 - (double)linear_errors / (dataset->num_samples * 0.1);
        
        printf("  • Puntos problematicos: %d\n", linear_errors);
        printf("  • Puntaje de separabilidad: %.2f/1.00\n", separability_score);
        
        if (separability_score > 0.8) {
            printf("  • ✅ Dataset probablemente linealmente separable\n");
            printf("  • Recomendacion: Prueba kernel lineal primero\n");
        } else if (separability_score > 0.5) {
            printf("  • ⚠️  Dataset parcialmente separable\n");
            printf("  • Recomendacion: Prueba kernel RBF\n");
        } else {
            printf("  • ❌ Dataset probablemente no linealmente separable\n");
            printf("  • Recomendacion: Usa kernel RBF o polinomial\n");
        }
    }
    
    // Sugerencias educativas
    if (learning_mode >= 1) {
        printf("\n🎓 EJERCICIOS SUGERIDOS:\n");
        printf("1. Intenta visualizar como cambiaria la distribucion si:\n");
        printf("   • Rotas los puntos 45 grados\n");
        printf("   • Escalas una caracteristica por 2\n");
        printf("   • Agregas ruido a los datos\n");
        
        printf("\n2. Preguntas para reflexionar:\n");
        printf("   • ¿Crees que un modelo lineal podria separar estas clases?\n");
        printf("   • ¿Donde estarian los vectores soporte mas probables?\n");
        printf("   • ¿Como afectaria el parametro C a la frontera de decision?\n");
    }
}

void print_feature_importance(SVM_Model* model, Dataset* dataset) {
    if (strcmp(model->kernel_type, "linear") != 0) {
        print_warning("La importancia de caracteristicas solo es directa para kernel lineal");
        return;
    }
    
    print_section("IMPORTANCIA DE CARACTERISTICAS");
    
    printf("Para SVM lineal, la importancia de cada caracteristica\n");
    printf("esta dada por el valor absoluto de su peso en el modelo.\n\n");
    
    // Calcular importancia relativa
    double total_importance = 0.0;
    double importances[MAX_FEATURES] = {0};
    
    for (int i = 0; i < model->num_features_trained; i++) {
        importances[i] = fabs(model->weights[i]);
        total_importance += importances[i];
    }
    
    // Ordenar caracteristicas por importancia
    int indices[MAX_FEATURES];
    for (int i = 0; i < model->num_features_trained; i++) indices[i] = i;
    
    for (int i = 0; i < model->num_features_trained - 1; i++) {
        for (int j = i + 1; j < model->num_features_trained; j++) {
            if (importances[indices[i]] < importances[indices[j]]) {
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
        }
    }
    
    printf("📊 TOP %d CARACTERISTICAS MAS IMPORTANTES:\n", model->num_features_trained < 10 ? model->num_features_trained : 10);
    printf("┌─────┬──────────────────────┬──────────────┬──────────────┬────────────────┐\n");
    printf("│ Pos │ Nombre               │      Peso    │  Importancia │  Porcentaje    │\n");
    printf("├─────┼──────────────────────┼──────────────┼──────────────┼────────────────┤\n");
    
    for (int i = 0; i < model->num_features_trained && i < 10; i++) {
        int idx = indices[i];
        double importance = importances[idx];
        double percentage = (total_importance > 0) ? (importance / total_importance * 100) : 0;
        
        printf("│ %3d │ %-20s │ %12.4f │ %12.4f │ %14.1f%% │\n",
               i + 1,
               dataset->feature_names[idx],
               model->weights[idx],
               importance,
               percentage);
    }
    printf("└─────┴──────────────────────┴──────────────┴──────────────┴────────────────┘\n");
    
    // Visualizacion de barras
    printf("\n📈 VISUALIZACION DE IMPORTANCIA:\n");
    
    for (int i = 0; i < model->num_features_trained && i < 8; i++) {
        int idx = indices[i];
        double percentage = (total_importance > 0) ? (importances[idx] / total_importance * 100) : 0;
        
        printf("\n%2d. %-20s ", i + 1, dataset->feature_names[idx]);
        
        int bar_length = (int)(percentage / 2);  // Escala para que quepa en terminal
        printf("[");
        for (int j = 0; j < bar_length; j++) {
            if (j < 10) printf("█");
            else if (j < 20) printf("▓");
            else if (j < 30) printf("▒");
            else printf("░");
        }
        for (int j = bar_length; j < 50; j++) printf(" ");
        printf("] %5.1f%%", percentage);
        
        // Indicar signo del peso
        if (model->weights[idx] > 0) {
            printf(" → Favorece clase +1");
        } else if (model->weights[idx] < 0) {
            printf(" → Favorece clase -1");
        }
    }
    
    // Interpretacion
    printf("\n\n🎯 INTERPRETACION:\n");
    printf("1. Peso positivo: La caracteristica favorece la clase +1\n");
    printf("2. Peso negativo: La caracteristica favorece la clase -1\n");
    printf("3. Magnitud: Que tan importante es la caracteristica\n");
    printf("4. Cercano a cero: La caracteristica tiene poco efecto\n");
    
    // Sugerencias basadas en importancia
    printf("\n💡 SUGERENCIAS:\n");
    
    // Identificar caracteristicas irrelevantes
    int irrelevant_features = 0;
    for (int i = 0; i < model->num_features_trained; i++) {
        if (fabs(model->weights[i]) < 0.01) irrelevant_features++;
    }
    
    if (irrelevant_features > 0) {
        printf("• %d caracteristicas tienen peso < 0.01\n", irrelevant_features);
        printf("• Considera eliminarlas para simplificar el modelo\n");
    }
    
    // Verificar si hay una caracteristica dominante
    double max_importance = 0.0;
    int dominant_idx = -1;
    for (int i = 0; i < model->num_features_trained; i++) {
        if (importances[i] > max_importance) {
            max_importance = importances[i];
            dominant_idx = i;
        }
    }
    
    if (max_importance / total_importance > 0.5 && dominant_idx != -1) {
        printf("• La caracteristica '%s' domina el modelo (%.1f%%)\n", 
               dataset->feature_names[dominant_idx],
               max_importance / total_importance * 100);
        printf("• Verifica que no haya leakage de datos o correlacion espuria\n");
    }
    
    // Ejercicio educativo
    if (learning_mode >= 1) {
        printf("\n🎓 EJERCICIO DE APRENDIZAJE:\n");
        printf("1. Entrena el modelo nuevamente eliminando la caracteristica\n");
        printf("   mas importante. ¿Como cambia la exactitud?\n");
        printf("2. Entrena solo con las 3 caracteristicas mas importantes.\n");
        printf("   ¿Pierdes mucha precision?\n");
        printf("3. ¿Que caracteristicas podrian estar correlacionadas?\n");
    }
}

void print_error_analysis(Dataset* dataset, SVM_Model* model) {
    print_section("ANALISIS DETALLADO DE ERRORES");
    
    // Calcular matriz de confusion
    int matrix[2][2] = {0};
    confusion_matrix_svm(model, dataset, matrix);
    
    // Identificar puntos mal clasificados
    DataPoint* false_positives[MAX_SAMPLES] = {0};
    DataPoint* false_negatives[MAX_SAMPLES] = {0};
    int fp_count = 0, fn_count = 0;
    
    for (int i = 0; i < dataset->num_samples; i++) {
        double prediction = predict_point(model, &dataset->points[i]);
        int predicted_label = (prediction >= 0) ? 1 : -1;
        
        if (predicted_label == 1 && dataset->points[i].label == -1) {
            false_positives[fp_count++] = &dataset->points[i];
        } else if (predicted_label == -1 && dataset->points[i].label == 1) {
            false_negatives[fn_count++] = &dataset->points[i];
        }
    }
    
    printf("📊 RESUMEN DE ERRORES:\n");
    printf("  • Total de muestras: %d\n", dataset->num_samples);
    printf("  • Clasificaciones correctas: %d (%.1f%%)\n", 
           matrix[0][0] + matrix[1][1],
           (double)(matrix[0][0] + matrix[1][1]) / dataset->num_samples * 100);
    printf("  • Errores totales: %d (%.1f%%)\n",
           matrix[1][0] + matrix[0][1],
           (double)(matrix[1][0] + matrix[0][1]) / dataset->num_samples * 100);
    
    printf("\n🔍 TIPOS DE ERROR:\n");
    printf("  • Falsos positivos (FP): %d\n", matrix[0][1]);
    printf("    (Clase -1 predicha como +1)\n");
    printf("  • Falsos negativos (FN): %d\n", matrix[1][0]);
    printf("    (Clase +1 predicha como -1)\n");
    
    // Analizar distancia al hiperplano de los errores
    printf("\n📏 DISTANCIA AL HIPERPLANO DE LOS ERRORES:\n");
    
    if (fp_count > 0) {
        double avg_fp_distance = 0.0;
        double min_fp_distance = DBL_MAX;
        for (int i = 0; i < fp_count; i++) {
            double dist = fabs(compute_distance_to_hyperplane(model, false_positives[i]));
            avg_fp_distance += dist;
            if (dist < min_fp_distance) min_fp_distance = dist;
        }
        avg_fp_distance /= fp_count;
        
        printf("  • Falsos positivos:\n");
        printf("    - Distancia promedio: %.4f\n", avg_fp_distance);
        printf("    - Distancia minima: %.4f\n", min_fp_distance);
        printf("    - Cercanos al margen: %.4f < distancia < %.4f\n", 
               min_fp_distance, avg_fp_distance);
    }
    
    if (fn_count > 0) {
        double avg_fn_distance = 0.0;
        double min_fn_distance = DBL_MAX;
        for (int i = 0; i < fn_count; i++) {
            double dist = fabs(compute_distance_to_hyperplane(model, false_negatives[i]));
            avg_fn_distance += dist;
            if (dist < min_fn_distance) min_fn_distance = dist;
        }
        avg_fn_distance /= fn_count;
        
        printf("  • Falsos negativos:\n");
        printf("    - Distancia promedio: %.4f\n", avg_fn_distance);
        printf("    - Distancia minima: %.4f\n", min_fn_distance);
        printf("    - Cercanos al margen: %.4f < distancia < %.4f\n",
               min_fn_distance, avg_fn_distance);
    }
    
    // Identificar patrones en los errores
    printf("\n🎯 PATRONES EN LOS ERRORES:\n");
    
    if (dataset->num_features >= 2 && (fp_count > 0 || fn_count > 0)) {
        printf("Ubicacion de errores en el espacio 2D:\n\n");
        
        // Crear mapa simple
        int grid_size = 30;
        char grid[grid_size][grid_size];
        for (int i = 0; i < grid_size; i++) {
            for (int j = 0; j < grid_size; j++) {
                grid[i][j] = ' ';
            }
        }
        
        double min_x = dataset->feature_min[0];
        double max_x = dataset->feature_max[0];
        double min_y = dataset->feature_min[1];
        double max_y = dataset->feature_max[1];
        
        // Marcar falsos positivos
        for (int i = 0; i < fp_count && i < 50; i++) {
            int x = (int)((false_positives[i]->features[0] - min_x) / (max_x - min_x) * (grid_size - 1));
            int y = (int)((false_positives[i]->features[1] - min_y) / (max_y - min_y) * (grid_size - 1));
            y = grid_size - 1 - y;
            
            if (x >= 0 && x < grid_size && y >= 0 && y < grid_size) {
                grid[y][x] = 'F';  // Falso positivo
            }
        }
        
        // Marcar falsos negativos
        for (int i = 0; i < fn_count && i < 50; i++) {
            int x = (int)((false_negatives[i]->features[0] - min_x) / (max_x - min_x) * (grid_size - 1));
            int y = (int)((false_negatives[i]->features[1] - min_y) / (max_y - min_y) * (grid_size - 1));
            y = grid_size - 1 - y;
            
            if (x >= 0 && x < grid_size && y >= 0 && y < grid_size) {
                grid[y][x] = 'N';  // Falso negativo
            }
        }
        
        // Dibujar hiperplano aproximado
        for (int i = 0; i < grid_size; i++) {
            for (int j = 0; j < grid_size; j++) {
                double x = min_x + (max_x - min_x) * j / (grid_size - 1);
                double y = min_y + (max_y - min_y) * (grid_size - 1 - i) / (grid_size - 1);
                
                DataPoint test = {0};
                test.features[0] = x;
                test.features[1] = y;
                
                double prediction = predict_point(model, &test);
                if (fabs(prediction) < 0.1 && grid[i][j] == ' ') {
                    grid[i][j] = '|';
                }
            }
        }
        
        // Imprimir grid
        printf("    ");
        for (int i = 0; i < grid_size; i++) {
            printf("%2d ", grid_size - i - 1);
            for (int j = 0; j < grid_size; j++) {
                char c = grid[i][j];
                if (c == 'F') printf("\033[41mF\033[0m");  // Rojo para FP
                else if (c == 'N') printf("\033[43mN\033[0m");  // Amarillo para FN
                else if (c == '|') printf("\033[32m|\033[0m");  // Verde para hiperplano
                else printf(" ");
            }
            printf("\n");
        }
        
        printf("\nLeyenda:\n");
        printf("  • \033[41mF\033[0m = Falso positivo (deberia ser -1, predicho +1)\n");
        printf("  • \033[43mN\033[0m = Falso negativo (deberia ser +1, predicho -1)\n");
        printf("  • \033[32m|\033[0m = Hiperplano de decision aproximado\n");
    }
    
    // Analisis de causas probables
    printf("\n🔎 CAUSAS PROBABLES DE ERRORES:\n");
    
    if (fp_count > fn_count * 2) {
        printf("  • Muchos mas falsos positivos que negativos\n");
        printf("  • Posible causa: El modelo es muy 'agresivo' prediciendo clase +1\n");
        printf("  • Solucion: Aumentar C para penalizar mas los errores de +1\n");
    } else if (fn_count > fp_count * 2) {
        printf("  • Muchos mas falsos negativos que positivos\n");
        printf("  • Posible causa: El modelo es muy 'conservador' prediciendo clase +1\n");
        printf("  • Solucion: Disminuir C o ajustar pesos de clases\n");
    }
    
    // Verificar si los errores estan cerca del hiperplano
    int errors_near_margin = 0;
    for (int i = 0; i < fp_count; i++) {
        double dist = fabs(compute_distance_to_hyperplane(model, false_positives[i]));
        if (dist < 0.5) errors_near_margin++;
    }
    for (int i = 0; i < fn_count; i++) {
        double dist = fabs(compute_distance_to_hyperplane(model, false_negatives[i]));
        if (dist < 0.5) errors_near_margin++;
    }
    
    double error_margin_ratio = (double)errors_near_margin / (fp_count + fn_count);
    
    if (error_margin_ratio > 0.7) {
        printf("  • La mayoria de errores estan cerca del hiperplano\n");
        printf("  • Esto es normal: son casos dificiles de clasificar\n");
        printf("  • El modelo esta haciendo un buen trabajo\n");
    } else if (error_margin_ratio < 0.3) {
        printf("  • Muchos errores lejos del hiperplano\n");
        printf("  • Posible causa: Frontera de decision incorrecta\n");
        printf("  • Solucion: Revisar kernel o parametros\n");
    }
    
    // Recomendaciones especificas
    printf("\n💡 RECOMENDACIONES PARA MEJORAR:\n");
    
    if (strcmp(model->kernel_type, "linear") == 0 && (fp_count + fn_count) > dataset->num_samples * 0.3) {
        printf("1. Cambia a kernel RBF (los datos pueden no ser lineales)\n");
    }
    
    if (model->C < 1.0 && errors_near_margin > 0) {
        printf("2. Aumenta el parametro C para penalizar mas los errores\n");
    }
    
    if (model->num_support_vectors > dataset->num_samples * 0.7) {
        printf("3. Demasiados vectores soporte - considera aumentar C\n");
    }
    
    // Ejercicio educativo
    if (learning_mode >= 1) {
        printf("\n🎓 EJERCICIO DE ANALISIS:\n");
        printf("1. Examina 3 errores especificos:\n");
        printf("   • ¿Por que el modelo se equivoco?\n");
        printf("   • ¿Estan cerca de otros puntos de diferente clase?\n");
        printf("   • ¿Son outliers o valores atipicos?\n");
        
        printf("\n2. Experimenta:\n");
        printf("   • Entrena con C mas alto y mas bajo\n");
        printf("   • Observa como cambian los errores\n");
        printf("   • ¿Mejora un tipo de error pero empeora otro?\n");
    }
}

void print_learning_curve(TrainingHistory* history) {
    if (history->iteration_count == 0) {
        print_warning("No hay datos de curva de aprendizaje");
        return;
    }
    
    print_section("CURVA DE APRENDIZAJE DEL MODELO");
    
    printf("La curva de aprendizaje muestra como evolucionan las metricas\n");
    printf("durante el entrenamiento del modelo SVM.\n\n");
    
    // Grafico combinado de error y exactitud
    int graph_height = 15;
    int graph_width = 60;
    
    // Encontrar maximos para escalar
    double max_error = 0.0;
    double max_accuracy = 0.0;
    for (int i = 0; i < history->iteration_count && i < MAX_ITERATIONS; i++) {
        if (history->error_history[i] > max_error) max_error = history->error_history[i];
        if (history->accuracy_history[i] > max_accuracy) max_accuracy = history->accuracy_history[i];
    }
    
    if (max_error < 0.001) max_error = 1.0;
    if (max_accuracy < 0.001) max_accuracy = 1.0;
    
    printf("Evolucion durante el entrenamiento:\n");
    printf("   Error (rojo)    Exactitud (verde)    Vectores soporte (amarillo)\n");
    printf("   ────────────    ─────────────────    ──────────────────────────\n");
    
    for (int h = graph_height; h >= 0; h--) {
        printf("%3.0f%% │", (double)h / graph_height * 100);
        
        for (int i = 0; i < history->iteration_count && i < graph_width; i++) {
            double error_norm = history->error_history[i] / max_error;
            double acc_norm = history->accuracy_history[i] / max_accuracy;
            double sv_norm = (double)history->sv_count_history[i] / current_dataset.num_samples;
            
            int error_pos = (int)(error_norm * graph_height);
            int acc_pos = (int)(acc_norm * graph_height);
            int sv_pos = (int)(sv_norm * graph_height);
            
            char symbol = ' ';
            if (h == error_pos && h == acc_pos && h == sv_pos) {
                symbol = '@';  // Cambiado de '★' a '@' para evitar warning
            } else if (h == error_pos && h == acc_pos) {
                symbol = 'X';
            } else if (h == error_pos && h == sv_pos) {
                symbol = 'E';
            } else if (h == acc_pos && h == sv_pos) {
                symbol = 'A';
            } else if (h == error_pos) {
                symbol = 'E';
            } else if (h == acc_pos) {
                symbol = 'A';
            } else if (h == sv_pos) {
                symbol = 'S';
            } else if (h == 0) {
                symbol = '_';
            }
            
            // Colorear segun simbolo
            switch(symbol) {
                case 'E': printf("\033[31mE\033[0m"); break;
                case 'A': printf("\033[32mA\033[0m"); break;
                case 'S': printf("\033[33mS\033[0m"); break;
                case 'X': printf("\033[35mX\033[0m"); break;
                case '@': printf("\033[36m@\033[0m"); break;
                case '_': printf("_"); break;
                default: printf(" ");
            }
        }
        printf("\n");
    }
    
    printf("     ");
    for (int i = 0; i < 6; i++) {
        printf("%-10d", (i+1) * (graph_width/6));
    }
    printf(" Iteraciones\n");
    
    // Analisis de la curva
    printf("\n📈 INTERPRETACION DE LA CURVA:\n");
    
    // Verificar convergencia
    int last_quarter = history->iteration_count * 3 / 4;
    double error_start = history->error_history[0];
    double error_end = history->error_history[history->iteration_count - 1];
    double error_change = (error_start - error_end) / error_start;
    
    if (error_change > 0.9) {
        printf("  • ✅ Convergencia rapida: Error reducido en >90%%\n");
    } else if (error_change > 0.5) {
        printf("  • ⚠️  Convergencia moderada: Error reducido en %.0f%%\n", error_change * 100);
    } else {
        printf("  • ❌ Convergencia lenta: Error reducido solo en %.0f%%\n", error_change * 100);
        printf("    Considera: Mas iteraciones, mayor tasa de aprendizaje\n");
    }
    
    // Verificar estabilidad
    double last_errors_avg = 0.0;
    for (int i = last_quarter; i < history->iteration_count; i++) {
        last_errors_avg += history->error_history[i];
    }
    last_errors_avg /= (history->iteration_count - last_quarter);
    
    double variance = 0.0;
    for (int i = last_quarter; i < history->iteration_count; i++) {
        double diff = history->error_history[i] - last_errors_avg;
        variance += diff * diff;
    }
    variance /= (history->iteration_count - last_quarter);
    
    if (variance < 0.0001) {
        printf("  • ✅ Estabilidad excelente: Error casi constante al final\n");
    } else if (variance < 0.001) {
        printf("  • ⚠️  Estabilidad buena: Pequenas fluctuaciones al final\n");
    } else {
        printf("  • ❌ Estabilidad pobre: Grandes fluctuaciones al final\n");
        printf("    Considera: Disminuir tasa de aprendizaje, mas datos\n");
    }
    
    // Analisis de sobreajuste basico
    double accuracy_start = history->accuracy_history[0];
    double accuracy_end = history->accuracy_history[history->iteration_count - 1];
    double accuracy_gain = accuracy_end - accuracy_start;
    
    if (accuracy_gain > 0.3) {
        printf("  • 📈 Gran mejora: Exactitud aumento en %.0f%% puntos\n", accuracy_gain * 100);
    } else if (accuracy_gain > 0.1) {
        printf("  • 📈 Mejora moderada: Exactitud aumento en %.0f%% puntos\n", accuracy_gain * 100);
    } else {
        printf("  • 📉 Mejora limitada: Exactitud aumento solo en %.0f%% puntos\n", accuracy_gain * 100);
    }
    
    // Evolucion de vectores soporte
    int sv_start = history->sv_count_history[0];
    int sv_end = history->sv_count_history[history->iteration_count - 1];
    
    printf("\n🔍 EVOLUCION DE VECTORES SOPORTE:\n");
    printf("  • Inicio: %d vectores soporte\n", sv_start);
    printf("  • Final: %d vectores soporte\n", sv_end);
    printf("  • Cambio: %+d (%.1f%%)\n", sv_end - sv_start, 
           (double)(sv_end - sv_start) / sv_start * 100);
    
    if (sv_end > sv_start * 1.5) {
        printf("  • ⚠️  Aumento significativo en vectores soporte\n");
        printf("    Podria indicar: Modelo mas complejo, menor margen\n");
    } else if (sv_end < sv_start * 0.5) {
        printf("  • ⚠️  Disminucion significativa en vectores soporte\n");
        printf("    Podria indicar: Modelo mas simple, mayor margen\n");
    }
    
    // Recomendaciones basadas en la curva
    printf("\n💡 RECOMENDACIONES BASADAS EN LA CURVA:\n");
    
    if (history->error_history[0] - history->error_history[history->iteration_count-1] < 0.1) {
        printf("1. Poca reduccion de error: Considera aumentar iteraciones\n");
    }
    
    if (variance > 0.01) {
        printf("2. Alta variabilidad: Disminuye tasa de aprendizaje\n");
    }
    
    if (accuracy_gain < 0.05 && history->iteration_count > 100) {
        printf("3. Poca mejora en exactitud: Revisa parametros o kernel\n");
    }
    
    // Ejercicio educativo
    if (learning_mode >= 1) {
        printf("\n🎓 EJERCICIO DE OBSERVACION:\n");
        printf("1. Identifica en que iteracion:\n");
        printf("   • El error comienza a estabilizarse\n");
        printf("   • La exactitud alcanza su maximo\n");
        printf("   • Los vectores soporte se estabilizan\n");
        
        printf("\n2. Preguntas para reflexionar:\n");
        printf("   • ¿El modelo necesitaba todas las iteraciones?\n");
        printf("   • ¿Hubo overfitting? (exactitud sube pero error no baja)\n");
        printf("   • ¿Como se relacionan error, exactitud y vectores soporte?\n");
    }
}

void print_confusion_matrix_visual(int tp, int tn, int fp, int fn) {
    print_section("MATRIZ DE CONFUSION VISUAL");
    
    printf("La matriz de confusion muestra el desempeno del modelo\n");
    printf("comparando las predicciones con las etiquetas reales.\n\n");
    
    // Matriz con colores
    printf("               PREDICCION DEL MODELO\n");
    printf("            ┌─────────────────────────────┐\n");
    printf("            │        +1        │    -1    │\n");
    printf("┌──────────┼──────────────────┼──────────┤\n");
    printf("│   REAL   │                  │          │\n");
    printf("│    +1    │"); 
    print_color(COLOR_GREEN, "   %4d TP     ", tp);
    printf("│");
    print_color(COLOR_RED, " %4d FN ", fn);
    printf("│\n");
    printf("├──────────┼──────────────────┼──────────┤\n");
    printf("│    -1    │");
    print_color(COLOR_RED, "   %4d FP     ", fp);
    printf("│");
    print_color(COLOR_GREEN, " %4d TN ", tn);
    printf("│\n");
    printf("└──────────┴──────────────────┴──────────┘\n");
    
    // Explicacion de cada celda
    printf("\n🔤 LEYENDA:\n");
    print_color(COLOR_GREEN, "  • TP (True Positive): Correctamente predicho como +1\n");
    print_color(COLOR_GREEN, "  • TN (True Negative): Correctamente predicho como -1\n");
    print_color(COLOR_RED,   "  • FP (False Positive): Incorrectamente predicho como +1 (deberia ser -1)\n");
    print_color(COLOR_RED,   "  • FN (False Negative): Incorrectamente predicho como -1 (deberia ser +1)\n");
    
    // Metricas derivadas
    printf("\n📊 METRICAS CALCULADAS:\n");
    
    double accuracy = (double)(tp + tn) / (tp + tn + fp + fn);
    double precision = (tp + fp > 0) ? (double)tp / (tp + fp) : 0.0;
    double recall = (tp + fn > 0) ? (double)tp / (tp + fn) : 0.0;
    double f1 = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0.0;
    
    printf("  • Exactitud (Accuracy): %.2f%%\n", accuracy * 100);
    printf("  • Precision: %.2f%%\n", precision * 100);
    printf("  • Recall (Sensibilidad): %.2f%%\n", recall * 100);
    printf("  • F1-Score: %.2f%%\n", f1 * 100);
    printf("  • Especificidad: %.2f%%\n", (double)tn / (tn + fp) * 100);
    
    // Interpretacion de metricas
    printf("\n🎯 INTERPRETACION DE METRICAS:\n");
    
    if (precision > 0.9 && recall > 0.9) {
        printf("  • ✅ Excelente desempeno en ambas clases\n");
    } else if (precision > 0.8 && recall > 0.8) {
        printf("  • ✅ Buen desempeno balanceado\n");
    } else if (precision > recall * 1.5) {
        printf("  • ⚠️  Alta precision pero bajo recall\n");
        printf("    El modelo es conservador con +1 (pocos FP, muchos FN)\n");
    } else if (recall > precision * 1.5) {
        printf("  • ⚠️  Alto recall pero baja precision\n");
        printf("    El modelo es agresivo con +1 (muchos FP, pocos FN)\n");
    }
    
    // Balance de clases
    double class_balance = (double)(tp + fn) / (tn + fp);
    if (class_balance > 2.0 || class_balance < 0.5) {
        printf("  • ⚠️  Desbalance de clases en los datos\n");
        printf("    Ratio +1/-1 en datos reales: %.2f:1\n", class_balance);
    }
    
    // Ejercicio educativo
    if (learning_mode >= 1) {
        printf("\n🎓 EJERCICIO DE INTERPRETACION:\n");
        printf("1. Para tu aplicacion, ¿que es mas importante?\n");
        printf("   • Evitar falsos positivos (alta precision)\n");
        printf("   • Capturar todos los positivos (alto recall)\n");
        printf("   • Balance general (alto F1-score)\n");
        
        printf("\n2. Experimenta:\n");
        printf("   • ¿Como cambia la matriz si aumentas C?\n");
        printf("   • ¿Y si usas un kernel diferente?\n");
        printf("   • ¿Que trade-offs observas?\n");
    }
}

// ============================ MANEJO DE DATASETS ============================

Dataset load_dataset(const char* filename) {
    Dataset dataset = {0};
    
    if (!filename || strlen(filename) == 0) {
        print_error("Nombre de archivo invalido");
        return dataset;
    }
    
    FILE* file = fopen(filename, "r");
    if (!file) {
        print_error("No se pudo abrir el archivo");
        return dataset;
    }
    
    char line[1024];
    int line_count = 0;
    int feature_count = 0;
    
    // Leer encabezado
    if (fgets(line, sizeof(line), file)) {
        line_count++;
        char* token = strtok(line, ",\n");
        feature_count = 0;
        
        while (token && feature_count < MAX_FEATURES) {
            strncpy(dataset.feature_names[feature_count], token, 49);
            dataset.feature_names[feature_count][49] = '\0';
            token = strtok(NULL, ",\n");
            feature_count++;
        }
        
        dataset.num_features = feature_count - 1;  // Ultima columna es la etiqueta
    }
    
    // Leer datos
    while (fgets(line, sizeof(line), file) && dataset.num_samples < MAX_SAMPLES) {
        char* token = strtok(line, ",\n");
        int col = 0;
        
        while (token && col <= dataset.num_features) {
            if (col < dataset.num_features) {
                dataset.points[dataset.num_samples].features[col] = atof(token);
                
                // Actualizar min/max
                if (dataset.num_samples == 0) {
                    dataset.feature_min[col] = dataset.feature_max[col] = 
                        dataset.points[dataset.num_samples].features[col];
                } else {
                    if (dataset.points[dataset.num_samples].features[col] < dataset.feature_min[col])
                        dataset.feature_min[col] = dataset.points[dataset.num_samples].features[col];
                    if (dataset.points[dataset.num_samples].features[col] > dataset.feature_max[col])
                        dataset.feature_max[col] = dataset.points[dataset.num_samples].features[col];
                }
            } else {
                // Ultima columna es la etiqueta
                dataset.points[dataset.num_samples].label = (atoi(token) > 0) ? 1 : -1;
            }
            
            token = strtok(NULL, ",\n");
            col++;
        }
        
        dataset.num_samples++;
    }
    
    fclose(file);
    
    // Generar nombre del dataset
    strncpy(dataset.name, filename, 99);
    dataset.name[99] = '\0';
    
    print_success("Dataset cargado exitosamente");
    printf("  • Muestras: %d\n", dataset.num_samples);
    printf("  • Caracteristicas: %d\n", dataset.num_features);
    
    return dataset;
}

Dataset create_linearly_separable_dataset(int samples) {
    Dataset dataset = {0};
    dataset.num_samples = samples;
    dataset.num_features = 2;
    
    strcpy(dataset.feature_names[0], "Caracteristica 1");
    strcpy(dataset.feature_names[1], "Caracteristica 2");
    strcpy(dataset.name, "Dataset Linealmente Separable");
    
    for (int i = 0; i < samples; i++) {
        // Generar puntos aleatorios
        dataset.points[i].features[0] = (double)rand() / RAND_MAX * 10.0 - 5.0;
        dataset.points[i].features[1] = (double)rand() / RAND_MAX * 10.0 - 5.0;
        
        // Separar linealmente: y = 2x + 1 + ruido
        double boundary = 2.0 * dataset.points[i].features[0] + 1.0;
        double noise = (double)rand() / RAND_MAX * 2.0 - 1.0;
        
        if (dataset.points[i].features[1] > boundary + noise) {
            dataset.points[i].label = 1;
        } else {
            dataset.points[i].label = -1;
        }
        
        // Actualizar min/max
        if (i == 0) {
            dataset.feature_min[0] = dataset.feature_max[0] = dataset.points[i].features[0];
            dataset.feature_min[1] = dataset.feature_max[1] = dataset.points[i].features[1];
        } else {
            if (dataset.points[i].features[0] < dataset.feature_min[0])
                dataset.feature_min[0] = dataset.points[i].features[0];
            if (dataset.points[i].features[0] > dataset.feature_max[0])
                dataset.feature_max[0] = dataset.points[i].features[0];
            if (dataset.points[i].features[1] < dataset.feature_min[1])
                dataset.feature_min[1] = dataset.points[i].features[1];
            if (dataset.points[i].features[1] > dataset.feature_max[1])
                dataset.feature_max[1] = dataset.points[i].features[1];
        }
    }
    
    return dataset;
}

Dataset create_xor_dataset(int samples) {
    Dataset dataset = {0};
    dataset.num_samples = samples;
    dataset.num_features = 2;
    
    strcpy(dataset.feature_names[0], "Caracteristica 1");
    strcpy(dataset.feature_names[1], "Caracteristica 2");
    strcpy(dataset.name, "Dataset XOR (No Lineal)");
    strcpy(dataset.description, "Problema XOR clasico, no separable linealmente");
    
    for (int i = 0; i < samples; i++) {
        // Generar puntos en cuadrantes
        int quadrant = rand() % 4;
        
        switch(quadrant) {
            case 0:  // Cuadrante I (+1)
                dataset.points[i].features[0] = (double)rand() / RAND_MAX * 2.0 + 1.0;
                dataset.points[i].features[1] = (double)rand() / RAND_MAX * 2.0 + 1.0;
                dataset.points[i].label = 1;
                break;
            case 1:  // Cuadrante II (-1)
                dataset.points[i].features[0] = (double)rand() / RAND_MAX * 2.0 - 3.0;
                dataset.points[i].features[1] = (double)rand() / RAND_MAX * 2.0 + 1.0;
                dataset.points[i].label = -1;
                break;
            case 2:  // Cuadrante III (+1)
                dataset.points[i].features[0] = (double)rand() / RAND_MAX * 2.0 - 3.0;
                dataset.points[i].features[1] = (double)rand() / RAND_MAX * 2.0 - 3.0;
                dataset.points[i].label = 1;
                break;
            case 3:  // Cuadrante IV (-1)
                dataset.points[i].features[0] = (double)rand() / RAND_MAX * 2.0 + 1.0;
                dataset.points[i].features[1] = (double)rand() / RAND_MAX * 2.0 - 3.0;
                dataset.points[i].label = -1;
                break;
        }
        
        // Actualizar min/max
        if (i == 0) {
            dataset.feature_min[0] = dataset.feature_max[0] = dataset.points[i].features[0];
            dataset.feature_min[1] = dataset.feature_max[1] = dataset.points[i].features[1];
        } else {
            if (dataset.points[i].features[0] < dataset.feature_min[0])
                dataset.feature_min[0] = dataset.points[i].features[0];
            if (dataset.points[i].features[0] > dataset.feature_max[0])
                dataset.feature_max[0] = dataset.points[i].features[0];
            if (dataset.points[i].features[1] < dataset.feature_min[1])
                dataset.feature_min[1] = dataset.points[i].features[1];
            if (dataset.points[i].features[1] > dataset.feature_max[1])
                dataset.feature_max[1] = dataset.points[i].features[1];
        }
    }
    
    return dataset;
}

void normalize_dataset(Dataset* dataset) {
    if (dataset->is_normalized || dataset->num_samples == 0) {
        return;
    }
    
    for (int i = 0; i < dataset->num_samples; i++) {
        for (int j = 0; j < dataset->num_features; j++) {
            // Normalizacion min-max a [0, 1]
            if (dataset->feature_max[j] - dataset->feature_min[j] > 0) {
                dataset->points[i].features[j] = 
                    (dataset->points[i].features[j] - dataset->feature_min[j]) /
                    (dataset->feature_max[j] - dataset->feature_min[j]);
            }
        }
    }
    
    // Actualizar min/max despues de normalizar
    for (int j = 0; j < dataset->num_features; j++) {
        dataset->feature_min[j] = 0.0;
        dataset->feature_max[j] = 1.0;
    }
    
    dataset->is_normalized = 1;
}

void print_dataset_info(Dataset* dataset) {
    if (dataset->num_samples == 0) {
        print_error("Dataset vacio");
        return;
    }
    
    print_section("INFORMACION DEL DATASET");
    
    printf("📋 Informacion basica:\n");
    printf("  • Nombre: %s\n", dataset->name);
    printf("  • Muestras: %d\n", dataset->num_samples);
    printf("  • Caracteristicas: %d\n", dataset->num_features);
    printf("  • Normalizado: %s\n", dataset->is_normalized ? "Si" : "No");
    
    if (strlen(dataset->description) > 0) {
        printf("  • Descripcion: %s\n", dataset->description);
    }
    
    // Distribucion de clases
    int class_pos = 0, class_neg = 0;
    for (int i = 0; i < dataset->num_samples; i++) {
        if (dataset->points[i].label == 1) class_pos++;
        else class_neg++;
    }
    
    printf("\n📊 Distribucion de clases:\n");
    printf("  • Clase +1: %d (%.1f%%)\n", class_pos, (double)class_pos/dataset->num_samples*100);
    printf("  • Clase -1: %d (%.1f%%)\n", class_neg, (double)class_neg/dataset->num_samples*100);
    
    // Estadisticas de caracteristicas
    printf("\n📐 Rango de caracteristicas:\n");
    for (int i = 0; i < dataset->num_features && i < 5; i++) {
        printf("  • %s: [%.4f, %.4f]\n", 
               dataset->feature_names[i],
               dataset->feature_min[i],
               dataset->feature_max[i]);
    }
    
    if (dataset->num_features > 5) {
        printf("  • ... y %d caracteristicas mas\n", dataset->num_features - 5);
    }
}

// ============================ FUNCIONES SVM ============================

double dot_product(double a[], double b[], int n) {
    double result = 0.0;
    for (int i = 0; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
}

double linear_kernel(double a[], double b[], int n) {
    return dot_product(a, b, n);
}

double rbf_kernel(double a[], double b[], int n, double gamma) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return exp(-gamma * sum);
}

double polynomial_kernel(double a[], double b[], int n, double degree) {
    double dot = dot_product(a, b, n);
    return pow(dot + 1.0, degree);
}

double predict_point(SVM_Model* model, DataPoint* point) {
    double result = 0.0;
    
    if (strcmp(model->kernel_type, "linear") == 0) {
        // f(x) = w·x + b
        result = dot_product(model->weights, point->features, model->num_features_trained) + model->bias;
    } else if (strcmp(model->kernel_type, "rbf") == 0) {
        // f(x) = Σ α_i·y_i·K(x, x_i) + b
        for (int i = 0; i < model->num_support_vectors; i++) {
            result += model->support_vectors[i].alpha * model->support_vectors[i].label *
                     rbf_kernel(point->features, model->support_vectors[i].features, 
                               model->num_features_trained, model->gamma);
        }
        result += model->bias;
    } else if (strcmp(model->kernel_type, "poly") == 0) {
        // f(x) = Σ α_i·y_i·K(x, x_i) + b
        for (int i = 0; i < model->num_support_vectors; i++) {
            result += model->support_vectors[i].alpha * model->support_vectors[i].label *
                     polynomial_kernel(point->features, model->support_vectors[i].features,
                                      model->num_features_trained, model->degree);
        }
        result += model->bias;
    }
    
    return result;
}

double compute_margin(SVM_Model* model) {
    // Para SVM lineal, margen = 1 / ||w||
    if (strcmp(model->kernel_type, "linear") == 0) {
        double norm_w = 0.0;
        for (int i = 0; i < model->num_features_trained; i++) {
            norm_w += model->weights[i] * model->weights[i];
        }
        norm_w = sqrt(norm_w);
        
        if (norm_w > 0) {
            return 1.0 / norm_w;
        }
    }
    
    // Para otros kernels, aproximamos basandonos en los vectores soporte
    if (model->num_support_vectors > 0) {
        double min_distance = DBL_MAX;
        for (int i = 0; i < model->num_support_vectors; i++) {
            double distance = fabs(compute_distance_to_hyperplane(model, &model->support_vectors[i]));
            if (distance < min_distance) {
                min_distance = distance;
            }
        }
        return min_distance;
    }
    
    return 0.0;
}

double compute_distance_to_hyperplane(SVM_Model* model, DataPoint* point) {
    // Distancia = |f(x)| / ||w|| para lineal
    // Para otros kernels, usamos f(x) directamente
    double prediction = predict_point(model, point);
    
    if (strcmp(model->kernel_type, "linear") == 0) {
        double norm_w = 0.0;
        for (int i = 0; i < model->num_features_trained; i++) {
            norm_w += model->weights[i] * model->weights[i];
        }
        norm_w = sqrt(norm_w);
        
        if (norm_w > 0) {
            return fabs(prediction) / norm_w;
        }
    }
    
    return fabs(prediction);
}

int is_support_vector(DataPoint* point, SVM_Model* model, double tolerance) {
    if (point->alpha > 0) {
        return 1;
    }
    
    // Verificar si esta cerca del hiperplano
    double distance = fabs(compute_distance_to_hyperplane(model, point));
    if (fabs(distance - 1.0) < tolerance) {
        return 1;
    }
    
    return 0;
}

// ============================ ENTRENAMIENTO ============================

void train_svm_linear(Dataset* dataset, SVM_Model* model) {
    print_section("ENTRENANDO SVM CON KERNEL LINEAL");
    
    // Inicializar modelo
    strcpy(model->kernel_type, "linear");
    model->num_features_trained = dataset->num_features;
    model->C = 1.0;
    model->iterations = 1000;
    
    // Inicializar pesos aleatoriamente
    for (int i = 0; i < model->num_features_trained; i++) {
        model->weights[i] = (double)rand() / RAND_MAX * 0.1 - 0.05;
    }
    model->bias = 0.0;
    
    // Entrenar con SGD
    clock_t start_time = clock();
    
    for (int iter = 0; iter < model->iterations; iter++) {
        double total_error = 0.0;
        int correct_predictions = 0;
        
        // Mezclar datos (simple shuffle)
        for (int i = dataset->num_samples - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            DataPoint temp = dataset->points[i];
            dataset->points[i] = dataset->points[j];
            dataset->points[j] = temp;
        }
        
        // Pasar por todos los puntos
        for (int i = 0; i < dataset->num_samples; i++) {
            double prediction = predict_point(model, &dataset->points[i]);
            double error = 1.0 - dataset->points[i].label * prediction;
            
            // Actualizar pesos si hay error
            if (error > 0) {
                double learning_rate = LEARNING_RATE * (1.0 - (double)iter / model->iterations);
                
                // Gradiente de hinge loss
                for (int j = 0; j < model->num_features_trained; j++) {
                    model->weights[j] += learning_rate * 
                        (dataset->points[i].label * dataset->points[i].features[j] - 
                         model->C * model->weights[j]);
                }
                model->bias += learning_rate * dataset->points[i].label;
                
                total_error += error;
            } else {
                correct_predictions++;
            }
            
            // Actualizar alpha (multiplicador de Lagrange)
            if (error > 0) {
                dataset->points[i].alpha += LEARNING_RATE;
                if (dataset->points[i].alpha > model->C) {
                    dataset->points[i].alpha = model->C;
                }
            } else if (dataset->points[i].alpha > 0) {
                dataset->points[i].alpha -= LEARNING_RATE;
                if (dataset->points[i].alpha < 0) {
                    dataset->points[i].alpha = 0;
                }
            }
        }
        
        // Guardar estadisticas de entrenamiento
        double avg_error = total_error / dataset->num_samples;
        double accuracy = (double)correct_predictions / dataset->num_samples;
        
        training_history.error_history[iter] = avg_error;
        training_history.accuracy_history[iter] = accuracy;
        
        // Identificar vectores soporte
        model->num_support_vectors = 0;
        for (int i = 0; i < dataset->num_samples && model->num_support_vectors < MAX_SAMPLES; i++) {
            if (is_support_vector(&dataset->points[i], model, 0.1)) {
                model->support_vectors[model->num_support_vectors] = dataset->points[i];
                model->num_support_vectors++;
            }
        }
        training_history.sv_count_history[iter] = model->num_support_vectors;
        
        // Calcular margen
        model->margin = compute_margin(model);
        training_history.margin_history[iter] = model->margin;
        
        // Mostrar progreso
        if (iter % 100 == 0 || iter == model->iterations - 1) {
            print_training_progress(model, iter, avg_error, accuracy);
        }
    }
    
    clock_t end_time = clock();
    model->training_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    // Calcular metricas finales
    model->training_accuracy = evaluate_svm(model, dataset);
    model->accuracy = model->training_accuracy;
    model->margin = compute_margin(model);
    model->trained_at = time(NULL);
    
    // Generar nombre automatico
    snprintf(model->name, sizeof(model->name), "SVM_Lineal_%.0f", model->training_accuracy * 100);
    
    print_success("Entrenamiento completado!");
    printf("  • Tiempo: %.2f segundos\n", model->training_time);
    printf("  • Exactitud: %.2f%%\n", model->accuracy * 100);
    printf("  • Vectores soporte: %d\n", model->num_support_vectors);
    printf("  • Margen: %.4f\n", model->margin);
}

double evaluate_svm(SVM_Model* model, Dataset* dataset) {
    if (dataset->num_samples == 0) {
        return 0.0;
    }
    
    int correct = 0;
    
    for (int i = 0; i < dataset->num_samples; i++) {
        double prediction = predict_point(model, &dataset->points[i]);
        int predicted_label = (prediction >= 0) ? 1 : -1;
        
        if (predicted_label == dataset->points[i].label) {
            correct++;
        }
    }
    
    return (double)correct / dataset->num_samples;
}

void confusion_matrix_svm(SVM_Model* model, Dataset* dataset, int matrix[2][2]) {
    // Inicializar matriz
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            matrix[i][j] = 0;
        }
    }
    
    // Calcular matriz de confusion
    for (int i = 0; i < dataset->num_samples; i++) {
        double prediction = predict_point(model, &dataset->points[i]);
        int predicted_label = (prediction >= 0) ? 1 : -1;
        int true_label = dataset->points[i].label;
        
        // Convertir etiquetas a indices (1->1, -1->0)
        int pred_idx = (predicted_label == 1) ? 1 : 0;
        int true_idx = (true_label == 1) ? 1 : 0;
        
        matrix[true_idx][pred_idx]++;
    }
}

// ============================ SISTEMA DE APRENDIZAJE ============================

void load_quiz_questions() {
    // Pregunta 1
    strcpy(quiz_questions[0].question, "¿Cual es el objetivo principal de SVM?");
    strcpy(quiz_questions[0].options[0], "Minimizar el error de entrenamiento");
    strcpy(quiz_questions[0].options[1], "Maximizar el margen entre clases");
    strcpy(quiz_questions[0].options[2], "Maximizar la complejidad del modelo");
    strcpy(quiz_questions[0].options[3], "Minimizar el numero de parametros");
    quiz_questions[0].correct_answer = 1;
    strcpy(quiz_questions[0].explanation, 
           "SVM busca MAXIMIZAR el margen entre las clases. Un margen mas grande "
           "generalmente lleva a mejor generalizacion y menor sobreajuste.");
    
    // Pregunta 2
    strcpy(quiz_questions[1].question, "¿Que son los vectores soporte en SVM?");
    strcpy(quiz_questions[1].options[0], "Todos los puntos de entrenamiento");
    strcpy(quiz_questions[1].options[1], "Los puntos mas alejados del hiperplano");
    strcpy(quiz_questions[1].options[2], "Los puntos mas cercanos al hiperplano");
    strcpy(quiz_questions[1].options[3], "Puntos generados aleatoriamente");
    quiz_questions[1].correct_answer = 2;
    strcpy(quiz_questions[1].explanation,
           "Los vectores soporte son los puntos de datos MAS CERCANOS al hiperplano. "
           "Son los unicos puntos que afectan la posicion del hiperplano.");
    
    // Pregunta 3
    strcpy(quiz_questions[2].question, "¿Que efecto tiene aumentar el parametro C en SVM?");
    strcpy(quiz_questions[2].options[0], "Aumenta el margen, permite mas errores");
    strcpy(quiz_questions[2].options[1], "Disminuye el margen, permite mas errores");
    strcpy(quiz_questions[2].options[2], "Aumenta el margen, penaliza mas errores");
    strcpy(quiz_questions[2].options[3], "Disminuye el margen, penaliza mas errores");
    quiz_questions[2].correct_answer = 3;
    strcpy(quiz_questions[2].explanation,
           "C alto → Margen pequeno, penaliza mucho los errores (riesgo overfit). "
           "C bajo → Margen grande, permite mas errores (riesgo underfit).");
    
    // Mas preguntas...
    total_questions = 3;  // Por ahora solo 3 preguntas
}

void ask_question(QuizQuestion* question) {
    print_section("PREGUNTA DE COMPRENSION");
    
    printf("%s\n\n", question->question);
    
    for (int i = 0; i < 4; i++) {
        printf("%d. %s\n", i + 1, question->options[i]);
    }
    
    printf("\nTu respuesta (1-4): ");
    int answer;
    scanf("%d", &answer);
    getchar();  // Limpiar buffer
    
    if (answer == question->correct_answer + 1) {
        quiz_score++;
        print_success("¡Correcto! 🎉");
    } else {
        print_error("Incorrecto. La respuesta correcta es: ");
        printf("%d. %s\n", question->correct_answer + 1, 
               question->options[question->correct_answer]);
    }
    
    printf("\n💡 Explicacion: %s\n", question->explanation);
    wait_for_enter();
}

void take_quiz() {
    print_header("EVALUACION DE CONOCIMIENTOS SVM");
    
    printf("Responde las siguientes preguntas para evaluar tu comprension.\n");
    printf("Cada pregunta vale 1 punto. ¡Buena suerte!\n\n");
    
    for (int i = 0; i < total_questions; i++) {
        ask_question(&quiz_questions[i]);
    }
    
    printf("\n📊 Resultados del quiz:\n");
    printf("  • Puntaje: %d/%d\n", quiz_score, total_questions);
    printf("  • Porcentaje: %.1f%%\n", (double)quiz_score / total_questions * 100);
    
    if ((double)quiz_score / total_questions >= 0.7) {
        print_success("¡Excelente! Dominas los conceptos basicos de SVM.");
    } else if ((double)quiz_score / total_questions >= 0.5) {
        print_warning("Buen trabajo, pero podrias repasar algunos conceptos.");
    } else {
        print_error("Necesitas estudiar mas los conceptos de SVM.");
    }
    
    wait_for_enter();
}

// ============================ MODOS DE INTERFAZ ============================

void interactive_mode() {
    int choice;
    
    do {
        clear_screen();
        print_header("MODO INTERACTIVO PRINCIPAL");
        
        printf("Selecciona una opcion:\n\n");
        printf("1. 🎓 Modo Aprendizaje\n");
        printf("2. 🏋️  Entrenar Modelo SVM\n");
        printf("3. 📊 Visualizar Modelo\n");
        printf("4. 📈 Analizar Desempeno\n");
        printf("5. 💾 Gestionar Modelos\n");
        printf("6. 📁 Gestionar Datasets\n");
        printf("7. 🧪 Modo Demo\n");
        printf("8. ❓ Ayuda\n");
        printf("9. 🚪 Salir\n");
        
        printf("\nOpcion: ");
        scanf("%d", &choice);
        getchar();
        
        switch(choice) {
            case 1:
                learning_mode_menu();
                break;
            case 2:
                training_mode();
                break;
            case 3:
                visualization_mode();
                break;
            case 4:
                analysis_mode();
                break;
            case 5:
                model_management_menu();
                break;
            case 6:
                dataset_management_menu();
                break;
            case 7:
                demo_mode();
                break;
            case 8:
                print_help();
                wait_for_enter();
                break;
            case 9:
                printf("\nSaliendo...\n");
                break;
            default:
                print_error("Opcion no valida");
                wait_for_enter();
        }
    } while (choice != 9);
}

void learning_mode_menu() {
    int choice;
    
    do {
        clear_screen();
        print_header("MODO APRENDIZAJE ACTIVO");
        
        printf("Selecciona una actividad:\n\n");
        printf("1. 📚 Tutorial Interactivo\n");
        printf("2. 🧠 Conceptos Teoricos\n");
        printf("3. 👁️  Visualizacion Paso a Paso\n");
        printf("4. ❓ Cuestionario de Evaluacion\n");
        printf("5. 🔍 Analisis de Casos Practicos\n");
        printf("6. 🏠 Volver al Menu Principal\n");
        
        printf("\nOpcion: ");
        scanf("%d", &choice);
        getchar();
        
        switch(choice) {
            case 1:
                interactive_tutorial();
                break;
            case 2:
                concept_explanation("svm_basics");
                break;
            case 3:
                step_by_step_training();
                break;
            case 4:
                take_quiz();
                break;
            case 5:
                explain_misconceptions(&current_model, &current_dataset);
                break;
            case 6:
                return;
            default:
                print_error("Opcion no valida");
                wait_for_enter();
        }
    } while (choice != 6);
}

void training_mode() {
    clear_screen();
    print_header("ENTRENAMIENTO DE MODELO SVM");
    
    if (current_dataset.num_samples == 0) {
        print_error("No hay dataset cargado");
        wait_for_enter();
        return;
    }
    
    printf("Selecciona el kernel a usar:\n\n");
    printf("1. 🔷 Lineal (datos linealmente separables)\n");
    printf("2. ⚪ RBF/Gaussiano (datos no lineales, recomendado)\n");
    printf("3. 🔶 Polinomial (relaciones polinomiales)\n");
    printf("4. 🏠 Volver\n");
    
    printf("\nOpcion: ");
    int choice;
    scanf("%d", &choice);
    getchar();
    
    if (choice == 4) return;
    
    // Configurar modelo segun eleccion
    switch(choice) {
        case 1:
            strcpy(current_model.kernel_type, "linear");
            current_model.C = 1.0;
            break;
        case 2:
            strcpy(current_model.kernel_type, "rbf");
            current_model.C = 1.0;
            current_model.gamma = 0.1;
            break;
        case 3:
            strcpy(current_model.kernel_type, "poly");
            current_model.C = 1.0;
            current_model.degree = 2.0;
            break;
        default:
            print_error("Opcion no valida");
            wait_for_enter();
            return;
    }
    
    // Entrenar modelo
    train_svm_linear(&current_dataset, &current_model);
    
    // Mostrar resultados
    print_model_info(&current_model);
    
    wait_for_enter();
}

void visualization_mode() {
    clear_screen();
    print_header("VISUALIZACION DEL MODELO");
    
    if (strlen(current_model.kernel_type) == 0) {
        print_error("No hay modelo entrenado");
        wait_for_enter();
        return;
    }
    
    printf("Selecciona tipo de visualizacion:\n\n");
    printf("1. 🎨 Visualizacion 2D Completa\n");
    printf("2. 📏 Hiperplano y Margen\n");
    printf("3. ⭐ Vectores Soporte\n");
    printf("4. 📈 Curva de Aprendizaje\n");
    printf("5. 🔢 Ecuacion del Modelo\n");
    printf("6. 🏠 Volver\n");
    
    printf("\nOpcion: ");
    int choice;
    scanf("%d", &choice);
    getchar();
    
    switch(choice) {
        case 1:
            print_svm_visualization(&current_dataset, &current_model);
            break;
        case 2:
            print_margin_visualization(&current_dataset, &current_model);
            break;
        case 3:
            print_support_vectors_visualization(&current_model);
            break;
        case 4:
            print_learning_curve(&training_history);
            break;
        case 5:
            print_hyperplane_equation(&current_model, &current_dataset);
            break;
        case 6:
            return;
        default:
            print_error("Opcion no valida");
    }
    
    wait_for_enter();
}

void demo_mode() {
    clear_screen();
    print_header("MODO DEMOSTRACION AUTOMATICA");
    
    printf("Este modo mostrara una demostracion completa de SVM.\n");
    printf("¿Comenzar demostracion? (s/n): ");
    
    char respuesta;
    scanf("%c", &respuesta);
    getchar();
    
    if (respuesta != 's' && respuesta != 'S') {
        return;
    }
    
    // Paso 1: Crear dataset
    print_section("PASO 1: CREANDO DATASET DE DEMOSTRACION");
    printf("Generando dataset XOR (no linealmente separable)...\n");
    current_dataset = create_xor_dataset(200);
    normalize_dataset(&current_dataset);
    print_dataset_info(&current_dataset);
    wait_for_enter();
    
    // Paso 2: Entrenar con kernel lineal
    print_section("PASO 2: ENTRENANDO CON KERNEL LINEAL");
    printf("Intentando separar datos no lineales con kernel lineal...\n");
    strcpy(current_model.kernel_type, "linear");
    current_model.C = 1.0;
    train_svm_linear(&current_dataset, &current_model);
    
    printf("\nObservacion: El kernel lineal no puede separar bien datos no lineales.\n");
    printf("Exactitud esperada: ~50%% (no mejor que adivinar)\n");
    wait_for_enter();
    
    // Paso 3: Entrenar con kernel RBF
    print_section("PASO 3: ENTRENANDO CON KERNEL RBF");
    printf("Usando kernel RBF para datos no lineales...\n");
    strcpy(current_model.kernel_type, "rbf");
    current_model.C = 1.0;
    current_model.gamma = 0.5;
    train_svm_linear(&current_dataset, &current_model);  // Reutilizando funcion
    
    printf("\nObservacion: El kernel RBF puede crear fronteras no lineales complejas.\n");
    printf("Exactitud esperada: >90%%\n");
    wait_for_enter();
    
    // Paso 4: Visualizar
    print_section("PASO 4: VISUALIZANDO RESULTADOS");
    printf("Mostrando frontera de decision del kernel RBF...\n");
    print_svm_visualization(&current_dataset, &current_model);
    wait_for_enter();
    
    // Paso 5: Analisis
    print_section("PASO 5: ANALISIS COMPARATIVO");
    printf("Comparacion de kernels:\n\n");
    print_kernel_comparison();
    
    printf("\n🎓 Conclusion de la demostracion:\n");
    printf("1. Kernel LINEAL: Solo para datos linealmente separables\n");
    printf("2. Kernel RBF: Puede manejar datos complejos no lineales\n");
    printf("3. Kernel POLINOMIAL: Para relaciones polinomiales conocidas\n");
    printf("4. RBF suele ser buen punto de partida\n");
    
    wait_for_enter();
}

void tutorial_mode() {
    clear_screen();
    print_header("TUTORIAL COMPLETO DE SVM");
    
    printf("Bienvenido al tutorial completo de Support Vector Machines.\n");
    printf("Este tutorial cubrira todos los conceptos paso a paso.\n\n");
    
    printf("¿Comenzar tutorial? (s/n): ");
    char respuesta;
    scanf("%c", &respuesta);
    getchar();
    
    if (respuesta != 's' && respuesta != 'S') {
        return;
    }
    
    // Tutorial paso a paso
    int step = 1;
    
    while (step <= 6) {
        clear_screen();
        char header_title[100];
        snprintf(header_title, sizeof(header_title), "TUTORIAL PASO %d/6", step);
        print_header(header_title);
        
        switch(step) {
            case 1:
                printf("📚 CONCEPTO 1: ¿Que es SVM?\n\n");
                printf("Support Vector Machine (SVM) es un algoritmo de aprendizaje\n");
                printf("supervisado usado para clasificacion y regresion.\n\n");
                printf("Caracteristicas clave:\n");
                printf("  • Busca el hiperplano que mejor separa las clases\n");
                printf("  • Maximiza el 'margen' entre clases\n");
                printf("  • Usa 'vectores soporte' (puntos mas cercanos)\n");
                printf("  • Puede manejar datos no lineales con 'kernels'\n");
                break;
                
            case 2:
                printf("📏 CONCEPTO 2: Hiperplano y Margen\n\n");
                printf("Hiperplano: Superficie de decision que separa las clases.\n");
                printf("En 2D: una linea, en 3D: un plano, en nD: hiperplano.\n\n");
                printf("Margen: Distancia entre el hiperplano y los vectores soporte.\n");
                printf("SVM busca MAXIMIZAR este margen para mejor generalizacion.\n");
                print_margin_visualization(&current_dataset, &current_model);
                break;
                
            case 3:
                printf("⭐ CONCEPTO 3: Vectores Soporte\n\n");
                printf("Son los puntos de datos MAS CERCANOS al hiperplano.\n");
                printf("Solo estos puntos afectan la posicion del hiperplano.\n\n");
                printf("Propiedades:\n");
                printf("  • Tienen α (alpha) > 0 (multiplicadores de Lagrange)\n");
                printf("  • Estan en el margen o dentro de el\n");
                printf("  • Determinan completamente el modelo\n");
                break;
                
            case 4:
                printf("🔄 CONCEPTO 4: Kernel Trick\n\n");
                printf("Problema: Datos no linealmente separables en espacio original.\n");
                printf("Solucion: Mapear a espacio de mayor dimension donde SI son separables.\n\n");
                printf("Kernels comunes:\n");
                printf("  • Lineal: K(x,y) = x·y\n");
                printf("  • RBF: K(x,y) = exp(-γ||x-y||²)\n");
                printf("  • Polinomial: K(x,y) = (x·y + c)^d\n");
                break;
                
            case 5:
                printf("⚙️  CONCEPTO 5: Parametro C\n\n");
                printf("C controla el trade-off entre:\n");
                printf("  • Margen grande (C pequeno)\n");
                printf("  • Pocos errores de clasificacion (C grande)\n\n");
                printf("Efectos:\n");
                printf("  • C muy alto: Sobreajuste (memoriza datos)\n");
                printf("  • C muy bajo: Subajuste (modelo muy simple)\n");
                printf("  • C optimo: Buen balance\n");
                break;
                
            case 6:
                printf("🎯 CONCEPTO 6: Aplicaciones Practicas\n\n");
                printf("SVM se usa en:\n");
                printf("  • Reconocimiento de imagenes\n");
                printf("  • Clasificacion de texto\n");
                printf("  • Bioinformatica\n");
                printf("  • Deteccion de fraudes\n");
                printf("  • Diagnostico medico\n\n");
                printf("Ventajas:\n");
                printf("  • Efectivo en espacios de alta dimension\n");
                printf("  • Eficiente en memoria (solo vectores soporte)\n");
                printf("  • Versatil (diferentes kernels)\n");
                break;
        }
        
        printf("\n[Enter] para continuar, [q] para salir: ");
        char input = getchar();
        if (input == 'q' || input == 'Q') {
            break;
        }
        
        step++;
    }
    
    printf("\n¡Tutorial completado! 🎉\n");
    wait_for_enter();
}

void analysis_mode() {
    clear_screen();
    print_header("ANALISIS DE DESEMPENO");
    
    if (strlen(current_model.kernel_type) == 0) {
        print_error("No hay modelo entrenado para analizar");
        wait_for_enter();
        return;
    }
    
    printf("Selecciona tipo de analisis:\n\n");
    printf("1. 📊 Metricas de Rendimiento\n");
    printf("2. 🔍 Analisis de Errores\n");
    printf("3. 📈 Importancia de Caracteristicas\n");
    printf("4. ⚠️  Diagnostico de Sobreajuste\n");
    printf("5. 💡 Sugerencias de Mejora\n");
    printf("6. 🏠 Volver\n");
    
    printf("\nOpcion: ");
    int choice;
    scanf("%d", &choice);
    getchar();
    
    switch(choice) {
        case 1: {
            // Calcular matriz de confusion
            int matrix[2][2] = {0};
            confusion_matrix_svm(&current_model, &current_dataset, matrix);
            
            print_confusion_matrix_visual(matrix[1][1], matrix[0][0], 
                                         matrix[0][1], matrix[1][0]);
            break;
        }
        case 2:
            print_error_analysis(&current_dataset, &current_model);
            break;
        case 3:
            print_feature_importance(&current_model, &current_dataset);
            break;
        case 4:
            // Diagnostico simple de sobreajuste
            printf("Diagnostico de sobreajuste:\n\n");
            printf("Exactitud entrenamiento: %.2f%%\n", 
                   current_model.training_accuracy * 100);
            
            if (current_model.validation_accuracy > 0) {
                printf("Exactitud validacion: %.2f%%\n", 
                       current_model.validation_accuracy * 100);
                
                double difference = current_model.training_accuracy - 
                                   current_model.validation_accuracy;
                
                if (difference > 0.15) {
                    print_error("⚠️  POSIBLE SOBREAJUSTE detectado");
                    printf("Diferencia > 15%%: %.1f%%\n", difference * 100);
                    printf("\nRecomendaciones:\n");
                    printf("1. Aumentar parametro C\n");
                    printf("2. Reducir complejidad del kernel\n");
                    printf("3. Conseguir mas datos de entrenamiento\n");
                } else if (difference > 0.05) {
                    print_warning("Leve posible sobreajuste");
                    printf("Diferencia: %.1f%%\n", difference * 100);
                } else {
                    print_success("Modelo bien balanceado");
                }
            } else {
                print_warning("No hay datos de validacion para comparar");
                printf("Usa train_svm_with_validation() para mejor diagnostico\n");
            }
            break;
        case 5:
            suggest_improvements(&current_model, &current_dataset);
            break;
        case 6:
            return;
        default:
            print_error("Opcion no valida");
    }
    
    wait_for_enter();
}

void model_management_menu() {
    int choice;
    
    do {
        clear_screen();
        print_header("GESTION DE MODELOS");
        
        printf("Selecciona una opcion:\n\n");
        printf("1. 💾 Guardar Modelo Actual\n");
        printf("2. 📂 Cargar Modelo\n");
        printf("3. 🖨️  Exportar Reporte\n");
        printf("4. ℹ️  Informacion del Modelo\n");
        printf("5. 🏠 Volver\n");
        
        printf("\nOpcion: ");
        scanf("%d", &choice);
    getchar();
        
        switch(choice) {
            case 1:
                save_model_interactive(&current_model);
                break;
            case 2:
                load_model_interactive(&current_model);
                break;
            case 3:
                export_full_report(&current_model, &current_dataset, "reporte_svm.txt");
                break;
            case 4:
                print_model_info(&current_model);
                wait_for_enter();
                break;
            case 5:
                return;
            default:
                print_error("Opcion no valida");
                wait_for_enter();
        }
    } while (choice != 5);
}

void dataset_management_menu() {
    int choice;
    
    do {
        clear_screen();
        print_header("GESTION DE DATASETS");
        
        printf("Selecciona una opcion:\n\n");
        printf("1. 📊 Informacion del Dataset\n");
        printf("2. 🎨 Visualizar Dataset\n");
        printf("3. 🔄 Generar Dataset de Prueba\n");
        printf("4. 💾 Guardar Dataset\n");
        printf("5. 🏠 Volver\n");
        
        printf("\nOpcion: ");
        scanf("%d", &choice);
        getchar();
        
        switch(choice) {
            case 1:
                print_dataset_info(&current_dataset);
                wait_for_enter();
                break;
            case 2:
                print_dataset_visualization(&current_dataset);
                wait_for_enter();
                break;
            case 3:
                printf("Selecciona tipo de dataset:\n");
                printf("1. Linealmente separable\n");
                printf("2. XOR (no lineal)\n");
                printf("3. Cancelar\n");
                
                int ds_choice;
                scanf("%d", &ds_choice);
                getchar();
                
                if (ds_choice == 1) {
                    current_dataset = create_linearly_separable_dataset(100);
                    normalize_dataset(&current_dataset);
                    print_success("Dataset lineal generado");
                } else if (ds_choice == 2) {
                    current_dataset = create_xor_dataset(100);
                    normalize_dataset(&current_dataset);
                    print_success("Dataset XOR generado");
                }
                wait_for_enter();
                break;
            case 4:
                save_dataset(&current_dataset, "dataset_actual.csv");
                break;
            case 5:
                return;
            default:
                print_error("Opcion no valida");
                wait_for_enter();
        }
    } while (choice != 5);
}

// ============================ FUNCIONES DE PERSISTENCIA ============================

int save_model(SVM_Model* model, const char* filename) {
    if (!filename || strlen(filename) == 0) {
        print_error("Nombre de archivo invalido");
        return 0;
    }
    
    FILE* file = fopen(filename, "wb");
    if (!file) {
        print_error("No se pudo crear el archivo");
        return 0;
    }
    
    // Escribir encabezado magico
    fprintf(file, "SVM_MODEL_V1.0\n");
    
    // Escribir informacion basica
    fprintf(file, "NAME:%s\n", model->name);
    fprintf(file, "KERNEL:%s\n", model->kernel_type);
    fprintf(file, "FEATURES:%d\n", model->num_features_trained);
    fprintf(file, "ACCURACY:%f\n", model->accuracy);
    fprintf(file, "MARGIN:%f\n", model->margin);
    fprintf(file, "C:%f\n", model->C);
    
    if (strcmp(model->kernel_type, "rbf") == 0) {
        fprintf(file, "GAMMA:%f\n", model->gamma);
    } else if (strcmp(model->kernel_type, "poly") == 0) {
        fprintf(file, "DEGREE:%f\n", model->degree);
    }
    
    // Escribir pesos
    fprintf(file, "WEIGHTS:");
    for (int i = 0; i < model->num_features_trained; i++) {
        fprintf(file, "%f", model->weights[i]);
        if (i < model->num_features_trained - 1) fprintf(file, ",");
    }
    fprintf(file, "\n");
    
    // Escribir bias
    fprintf(file, "BIAS:%f\n", model->bias);
    
    // Escribir fecha
    fprintf(file, "TRAINED_AT:%ld\n", model->trained_at);
    
    fclose(file);
    
    print_success("Modelo guardado exitosamente");
    return 1;
}

int load_model(SVM_Model* model, const char* filename) {
    if (!filename || strlen(filename) == 0) {
        print_error("Nombre de archivo invalido");
        return 0;
    }
    
    FILE* file = fopen(filename, "rb");
    if (!file) {
        print_error("No se pudo abrir el archivo");
        return 0;
    }
    
    char line[1024];
    
    // Leer encabezado magico
    if (!fgets(line, sizeof(line), file) || strstr(line, "SVM_MODEL") == NULL) {
        fclose(file);
        print_error("Formato de archivo invalido");
        return 0;
    }
    
    // Inicializar modelo
    memset(model, 0, sizeof(SVM_Model));
    
    // Leer parametros
    while (fgets(line, sizeof(line), file)) {
        line[strcspn(line, "\n")] = 0;  // Eliminar newline
        
        if (strncmp(line, "NAME:", 5) == 0) {
            strcpy(model->name, line + 5);
        } else if (strncmp(line, "KERNEL:", 7) == 0) {
            strcpy(model->kernel_type, line + 7);
        } else if (strncmp(line, "FEATURES:", 9) == 0) {
            model->num_features_trained = atoi(line + 9);
        } else if (strncmp(line, "ACCURACY:", 9) == 0) {
            model->accuracy = atof(line + 9);
        } else if (strncmp(line, "MARGIN:", 7) == 0) {
            model->margin = atof(line + 7);
        } else if (strncmp(line, "C:", 2) == 0) {
            model->C = atof(line + 2);
        } else if (strncmp(line, "GAMMA:", 6) == 0) {
            model->gamma = atof(line + 6);
        } else if (strncmp(line, "DEGREE:", 7) == 0) {
            model->degree = atof(line + 7);
        } else if (strncmp(line, "WEIGHTS:", 8) == 0) {
            char* token = strtok(line + 8, ",");
            int i = 0;
            while (token && i < model->num_features_trained) {
                model->weights[i++] = atof(token);
                token = strtok(NULL, ",");
            }
        } else if (strncmp(line, "BIAS:", 5) == 0) {
            model->bias = atof(line + 5);
        } else if (strncmp(line, "TRAINED_AT:", 11) == 0) {
            model->trained_at = atol(line + 11);
        }
    }
    
    fclose(file);
    
    // Actualizar metricas globales
    current_metrics.train_accuracy = model->accuracy;
    
    return 1;
}

void save_model_interactive(SVM_Model* model) {
    if (strlen(model->kernel_type) == 0) {
        print_error("No hay modelo entrenado para guardar");
        wait_for_enter();
        return;
    }
    
    printf("Nombre del archivo para guardar (ej: modelo.svm): ");
    char filename[256];
    scanf("%255s", filename);
    getchar();
    
    if (save_model(model, filename)) {
        strcpy(current_model_file, filename);
    }
    
    wait_for_enter();
}

void load_model_interactive(SVM_Model* model) {
    printf("Nombre del archivo a cargar: ");
    char filename[256];
    scanf("%255s", filename);
    getchar();
    
    if (load_model(model, filename)) {
        strcpy(current_model_file, filename);
        print_model_info(model);
    }
    
    wait_for_enter();
}

void save_dataset(Dataset* dataset, const char* filename) {
    if (!filename || dataset->num_samples == 0) {
        print_error("Datos invalidos para guardar");
        return;
    }
    
    FILE* file = fopen(filename, "w");
    if (!file) {
        print_error("No se pudo crear el archivo");
        return;
    }
    
    // Escribir encabezado
    for (int i = 0; i < dataset->num_features; i++) {
        fprintf(file, "%s", dataset->feature_names[i]);
        if (i < dataset->num_features - 1) fprintf(file, ",");
        else fprintf(file, ",label\n");
    }
    
    // Escribir datos
    for (int i = 0; i < dataset->num_samples; i++) {
        for (int j = 0; j < dataset->num_features; j++) {
            fprintf(file, "%f", dataset->points[i].features[j]);
            if (j < dataset->num_features - 1) fprintf(file, ",");
        }
        fprintf(file, ",%d\n", dataset->points[i].label);
    }
    
    fclose(file);
    
    print_success("Dataset guardado exitosamente");
    printf("Archivo: %s\n", filename);
    printf("Muestras: %d\n", dataset->num_samples);
    
    wait_for_enter();
}

void export_full_report(SVM_Model* model, Dataset* dataset, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        print_error("No se pudo crear el archivo de reporte");
        return;
    }
    
    fprintf(file, "========================================\n");
    fprintf(file, "        REPORTE COMPLETO SVM\n");
    fprintf(file, "========================================\n\n");
    
    fprintf(file, "Fecha de generacion: %s\n", ctime(&model->trained_at));
    
    // Informacion del modelo
    fprintf(file, "\n1. INFORMACION DEL MODELO:\n");
    fprintf(file, "   • Nombre: %s\n", model->name);
    fprintf(file, "   • Kernel: %s\n", model->kernel_type);
    fprintf(file, "   • Exactitud: %.2f%%\n", model->accuracy * 100);
    fprintf(file, "   • Margen: %.4f\n", model->margin);
    fprintf(file, "   • Vectores soporte: %d\n", model->num_support_vectors);
    fprintf(file, "   • Parametro C: %.4f\n", model->C);
    
    if (strcmp(model->kernel_type, "rbf") == 0) {
        fprintf(file, "   • Gamma: %.4f\n", model->gamma);
    } else if (strcmp(model->kernel_type, "poly") == 0) {
        fprintf(file, "   • Grado: %.0f\n", model->degree);
    }
    
    // Informacion del dataset
    fprintf(file, "\n2. INFORMACION DEL DATASET:\n");
    fprintf(file, "   • Muestras: %d\n", dataset->num_samples);
    fprintf(file, "   • Caracteristicas: %d\n", dataset->num_features);
    
    int class_pos = 0, class_neg = 0;
    for (int i = 0; i < dataset->num_samples; i++) {
        if (dataset->points[i].label == 1) class_pos++;
        else class_neg++;
    }
    fprintf(file, "   • Clase +1: %d (%.1f%%)\n", class_pos, (double)class_pos/dataset->num_samples*100);
    fprintf(file, "   • Clase -1: %d (%.1f%%)\n", class_neg, (double)class_neg/dataset->num_samples*100);
    
    // Ecuacion del modelo
    fprintf(file, "\n3. ECUACION DEL MODELO:\n");
    if (strcmp(model->kernel_type, "linear") == 0) {
        fprintf(file, "   f(x) = ");
        int terms_printed = 0;
        for (int i = 0; i < dataset->num_features && i < 6; i++) {
            if (fabs(model->weights[i]) > 0.0001) {
                if (terms_printed > 0 && model->weights[i] >= 0) fprintf(file, " + ");
                else if (model->weights[i] < 0) fprintf(file, " - ");
                
                fprintf(file, "%.4f·%s", fabs(model->weights[i]), dataset->feature_names[i]);
                terms_printed++;
            }
        }
        
        if (fabs(model->bias) > 0.0001) {
            if (model->bias >= 0) fprintf(file, " + %.4f", model->bias);
            else fprintf(file, " - %.4f", -model->bias);
        }
        fprintf(file, "\n");
    }
    
    // Metricas de rendimiento
    fprintf(file, "\n4. METRICAS DE RENDIMIENTO:\n");
    
    int matrix[2][2] = {0};
    confusion_matrix_svm(model, dataset, matrix);
    
    fprintf(file, "   • Matriz de confusion:\n");
    fprintf(file, "         Prediccion\n");
    fprintf(file, "        +1     -1\n");
    fprintf(file, "     +1 %4d  %4d\n", matrix[1][1], matrix[1][0]);
    fprintf(file, "Real\n");
    fprintf(file, "     -1 %4d  %4d\n", matrix[0][1], matrix[0][0]);
    
    double accuracy = (double)(matrix[0][0] + matrix[1][1]) / dataset->num_samples;
    double precision = (matrix[1][1] + matrix[0][1] > 0) ? 
                      (double)matrix[1][1] / (matrix[1][1] + matrix[0][1]) : 0.0;
    double recall = (matrix[1][1] + matrix[1][0] > 0) ? 
                   (double)matrix[1][1] / (matrix[1][1] + matrix[1][0]) : 0.0;
    double f1 = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0.0;
    
    fprintf(file, "   • Metricas derivadas:\n");
    fprintf(file, "     - Exactitud: %.2f%%\n", accuracy * 100);
    fprintf(file, "     - Precision: %.2f%%\n", precision * 100);
    fprintf(file, "     - Recall:    %.2f%%\n", recall * 100);
    fprintf(file, "     - F1-Score:  %.2f%%\n", f1 * 100);
    
    // Recomendaciones
    fprintf(file, "\n5. RECOMENDACIONES:\n");
    
    if (strcmp(model->kernel_type, "linear") == 0 && model->accuracy < 0.7) {
        fprintf(file, "   • Considera probar kernel RBF para mejor rendimiento\n");
    }
    
    if (model->C < 0.1) {
        fprintf(file, "   • Considera aumentar C para permitir mas errores\n");
    }
    
    if (model->margin < 0.1) {
        fprintf(file, "   • Margen muy pequeno, riesgo de sobreajuste\n");
    }
    
    fprintf(file, "\n========================================\n");
    fprintf(file, "        FIN DEL REPORTE\n");
    fprintf(file, "========================================\n");
    
    fclose(file);
    
    print_success("Reporte generado exitosamente");
    printf("Archivo: %s\n", filename);
    
    wait_for_enter();
}

// ============================ FUNCIONES RESTANTES ============================

void interactive_tutorial() {
    clear_screen();
    print_header("TUTORIAL INTERACTIVO DE SVM");
    
    printf("Este tutorial te guiara paso a paso en el entrenamiento\n");
    printf("y comprension de un modelo SVM.\n\n");
    
    printf("Vamos a crear un dataset simple y entrenar un modelo.\n");
    wait_for_enter();
    
    // Crear dataset
    printf("1. Creando dataset linealmente separable...\n");
    current_dataset = create_linearly_separable_dataset(50);
    normalize_dataset(&current_dataset);
    print_dataset_visualization(&current_dataset);
    wait_for_enter();
    
    printf("2. Entrenando modelo SVM lineal...\n");
    strcpy(current_model.kernel_type, "linear");
    train_svm_linear(&current_dataset, &current_model);
    wait_for_enter();
    
    printf("3. Visualizando resultado...\n");
    print_svm_visualization(&current_dataset, &current_model);
    wait_for_enter();
    
    printf("4. Analizando vectores soporte...\n");
    print_support_vectors_visualization(&current_model);
    wait_for_enter();
    
    printf("5. Evaluando modelo...\n");
    print_confusion_matrix_visual(
        current_metrics.true_positives,
        current_metrics.true_negatives,
        current_metrics.false_positives,
        current_metrics.false_negatives
    );
    
    printf("\n🎓 Tutorial completado!\n");
    printf("Has aprendido:\n");
    printf("  • Como se entrena un SVM\n");
    printf("  • Que son los vectores soporte\n");
    printf("  • Como se visualiza un modelo\n");
    printf("  • Como se evalua el rendimiento\n");
    
    wait_for_enter();
}

void step_by_step_training() {
    clear_screen();
    print_header("ENTRENAMIENTO PASO A PASO");
    
    if (current_dataset.num_samples == 0) {
        print_error("No hay dataset cargado");
        wait_for_enter();
        return;
    }
    
    printf("Este modo mostrara el entrenamiento iteracion por iteracion.\n");
    printf("¿Comenzar? (s/n): ");
    
    char respuesta;
    scanf("%c", &respuesta);
    getchar();
    
    if (respuesta != 's' && respuesta != 'S') {
        return;
    }
    
    // Configurar para modo paso a paso
    learning_mode = 2;
    train_svm_linear(&current_dataset, &current_model);
    learning_mode = 0;
    
    printf("\n🏁 Entrenamiento completado!\n");
    printf("Observaciones:\n");
    printf("  • El error disminuye gradualmente\n");
    printf("  • La exactitud aumenta\n");
    printf("  • Los vectores soporte se estabilizan\n");
    printf("  • El margen converge a un valor optimo\n");
    
    wait_for_enter();
}

void concept_explanation(const char* concept) {
    clear_screen();
    
    if (strcmp(concept, "svm_basics") == 0) {
        print_header("CONCEPTOS BASICOS DE SVM");
        
        printf("📚 TEORIA FUNDAMENTAL:\n\n");
        
        printf("1. HIPERPLANO:\n");
        printf("   • Superficie de decision que separa clases\n");
        printf("   • En 2D: linea recta (w·x + b = 0)\n");
        printf("   • En nD: hiperplano n-dimensional\n\n");
        
        printf("2. MARGEN:\n");
        printf("   • Distancia entre el hiperplano y puntos mas cercanos\n");
        printf("   • SVM busca MAXIMIZAR este margen\n");
        printf("   • Margen grande = mejor generalizacion\n\n");
        
        printf("3. VECTORES SOPORTE:\n");
        printf("   • Puntos que estan en el margen\n");
        printf("   • Determinan completamente el hiperplano\n");
        printf("   • Tienen α (alpha) > 0\n\n");
        
        printf("4. FORMULACION MATEMATICA:\n");
        printf("   Problema primal: Minimizar ½||w||² + C·Σξ_i\n");
        printf("   Sujeto a: y_i(w·x_i + b) ≥ 1 - ξ_i, ξ_i ≥ 0\n\n");
        
        printf("5. KERNEL TRICK:\n");
        printf("   • Transforma datos a espacio de mayor dimension\n");
        printf("   • Donde se vuelven linealmente separables\n");
        printf("   • Sin calcular explicitamente la transformacion\n");
        
    } else if (strcmp(concept, "kernels") == 0) {
        print_kernel_comparison();
    }
    
    wait_for_enter();
}

void explain_misconceptions(SVM_Model* model, Dataset* dataset) {
    print_header("ANALISIS DE ERRORES COMUNES");
    
    printf("1. ERROR: SVM solo funciona con datos linealmente separables\n");
    printf("   REALIDAD: Con kernels (RBF, polinomial) puede manejar datos complejos\n\n");
    
    printf("2. ERROR: Mas vectores soporte siempre es mejor\n");
    printf("   REALIDAD: Pocos vectores soporte pueden indicar buen margen\n");
    printf("            Muchos pueden indicar sobreajuste\n\n");
    
    printf("3. ERROR: C alto siempre da mejor exactitud\n");
    printf("   REALIDAD: C alto puede causar sobreajuste\n");
    printf("            C bajo puede causar subajuste\n");
    printf("            Necesita balance\n\n");
    
    printf("4. ERROR: SVM es lento con muchos datos\n");
    printf("   REALIDAD: Depende de la implementacion\n");
    printf("            Solo vectores soporte se usan en prediccion\n");
    printf("            Eficiente en memoria\n\n");
    
    printf("5. ERROR: RBF siempre es mejor que lineal\n");
    printf("   REALIDAD: Lineal es mas rapido e interpretable\n");
    printf("            RBF para datos complejos no lineales\n");
    printf("            Empezar simple (lineal), luego complejo (RBF)\n");
    
    wait_for_enter();
}

void suggest_improvements(SVM_Model* model, Dataset* dataset) {
    print_header("SUGERENCIAS PARA MEJORAR EL MODELO");
    
    printf("Basado en el analisis del modelo actual:\n\n");
    
    if (strcmp(model->kernel_type, "linear") == 0 && model->accuracy < 0.7) {
        printf("1. CAMBIAR KERNEL:\n");
        printf("   • Exactitud actual: %.1f%%\n", model->accuracy * 100);
        printf("   • Considera probar kernel RBF\n");
        printf("   • Los datos podrian no ser linealmente separables\n\n");
    }
    
    if (model->margin < 0.1) {
        printf("2. AJUSTAR PARAMETRO C:\n");
        printf("   • Margen actual: %.4f (muy pequeno)\n", model->margin);
        printf("   • Considera disminuir C\n");
        printf("   • Esto aumentara el margen\n\n");
    } else if (model->margin > 2.0) {
        printf("2. AJUSTAR PARAMETRO C:\n");
        printf("   • Margen actual: %.4f (muy grande)\n", model->margin);
        printf("   • Considera aumentar C\n");
        printf("   • Esto reducira el margen pero mejorara clasificacion\n\n");
    }
    
    if (model->num_support_vectors > dataset->num_samples * 0.7) {
        printf("3. REDUCIR COMPLEJIDAD:\n");
        printf("   • Vectores soporte: %d/%d (%.1f%%)\n", 
               model->num_support_vectors, dataset->num_samples,
               (double)model->num_support_vectors/dataset->num_samples*100);
        printf("   • Muchos vectores soporte indican posible sobreajuste\n");
        printf("   • Considera aumentar C o cambiar kernel\n\n");
    }
    
    if (dataset->num_samples < 100) {
        printf("4. CONSEGUIR MAS DATOS:\n");
        printf("   • Muestras actuales: %d\n", dataset->num_samples);
        printf("   • Considera aumentar el dataset\n");
        printf("   • Mas datos mejoran generalizacion\n\n");
    }
    
    printf("🎯 ESTRATEGIA RECOMENDADA:\n");
    printf("1. Dividir datos en entrenamiento/validacion/prueba\n");
    printf("2. Probar diferentes kernels\n");
    printf("3. Ajustar hiperparametros con validacion cruzada\n");
    printf("4. Evaluar en conjunto de prueba independiente\n");
    
    wait_for_enter();
}

void draw_box(const char* title, const char* content) {
    int width = terminal_width - 4;
    printf("┌─");
    for (int i = 0; i < width; i++) printf("─");
    printf("─┐\n");
    
    printf("│ %-*s │\n", width, title);
    
    printf("├─");
    for (int i = 0; i < width; i++) printf("─");
    printf("─┤\n");
    
    // Imprimir contenido con wrap
    char buffer[1024];
    strncpy(buffer, content, sizeof(buffer)-1);
    char* token = strtok(buffer, "\n");
    while (token) {
        printf("│ %-*s │\n", width, token);
        token = strtok(NULL, "\n");
    }
    
    printf("└─");
    for (int i = 0; i < width; i++) printf("─");
    printf("─┘\n");
}

void animate_progress(const char* message, int duration_ms) {
    printf("%s ", message);
    fflush(stdout);
    
    int steps = duration_ms / 100;
    for (int i = 0; i < steps; i++) {
        printf(".");
        fflush(stdout);
        usleep(100000);  // 100ms
    }
    printf(" ✓\n");
}
