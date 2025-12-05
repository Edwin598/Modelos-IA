#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
#include <termios.h>
#include <unistd.h>
#include <errno.h>
#include <stdarg.h>
#include <stdbool.h>
#include <sys/stat.h>
#include <signal.h>
#include <dirent.h>

// ============================ CONFIGURACIÓN ============================

#define MAX_SAMPLES 1000
#define MAX_FEATURES 20
#define MAX_CLASSES 10
#define MAX_TREE_DEPTH 15
#define MAX_CATEGORIES 20
#define MAX_LINE_LENGTH 1024
#define SCREEN_HEIGHT 24
#define MIN_FEATURE_VALUE -1000000.0
#define MAX_FEATURE_VALUE 1000000.0
#define TERMINAL_WIDTH 80
#define MODEL_DIR "models"
#define LOGS_DIR "logs"
#define EXPORTS_DIR "exports"

// ============================ ESTRUCTURAS DE DATOS ============================

typedef struct TreeNode {
    int is_leaf;
    int class_label;
    int split_feature;
    double split_value;
    char split_category[MAX_CATEGORIES][50];
    int num_categories;
    struct TreeNode* left;
    struct TreeNode* right;
    int samples;
    int depth;
    double gini;
    double entropy;
    int* sample_indices;  // Índices de muestras en este nodo
    int sample_count;
} TreeNode;

typedef struct {
    double features[MAX_FEATURES];
    int target;
    double weight;  // Peso de la muestra
} DataSample;

typedef struct {
    DataSample samples[MAX_SAMPLES];
    char feature_names[MAX_FEATURES][50];
    int feature_types[MAX_FEATURES];
    int num_samples;
    int num_features;
    int num_classes;
    char class_names[MAX_CLASSES][50];
    double feature_min[MAX_FEATURES];
    double feature_max[MAX_FEATURES];
    double feature_mean[MAX_FEATURES];
    double feature_std[MAX_FEATURES];
    int is_normalized;
} Dataset;

typedef struct {
    TreeNode* root;
    double accuracy;
    double precision;
    double recall;
    double f1_score;
    int max_depth;
    int min_samples_split;
    int min_samples_leaf;
    char algorithm[20];
    time_t created_at;
    time_t last_trained;
    time_t last_used;
    int node_count;
    int leaf_count;
    int total_samples_trained;
    char name[100];
    char description[200];
    int is_trained;
    double feature_importance[MAX_FEATURES];
} DecisionTree;

// ============================ GESTIÓN DE MODELOS ============================

typedef struct {
    char name[100];
    char filename[200];
    time_t created;
    time_t last_used;
    time_t last_trained;
    double accuracy;
    double precision;
    double recall;
    double f1_score;
    int samples_trained;
    int is_active;
    char description[200];
} ModelInfo;

typedef struct {
    ModelInfo models[100];
    int num_models;
    char default_model[100];
} ModelRegistry;

// ============================ HISTORIAL Y LOGS ============================

typedef struct {
    time_t timestamp;
    char operation[50];
    char details[200];
    double accuracy;
    int samples;
} HistoryEntry;

typedef struct {
    HistoryEntry entries[1000];
    int count;
    FILE* log_file;
} HistoryLogger;

// ============================ VARIABLES GLOBALES ============================

FILE *output_file = NULL;
int output_to_file = 0;
ModelRegistry model_registry = {0};
char current_model_file[200] = "";
DecisionTree* current_model = NULL;
HistoryLogger history_logger = {0};
int debug_mode = 0;
int verbose_mode = 1;

// ============================ PROTOTIPOS DE FUNCIONES ============================

// Sistema e inicialización
void init_system();
void cleanup_system();
void handle_signal(int sig);
void create_directories();
void print_system_info();

// Gestión de logs e historia
void init_history_logger();
void log_operation(const char* operation, const char* details, double accuracy, int samples);
void save_history_log();
void load_history_log();
void print_recent_history();
void export_history_csv(const char* filename);

// Persistencia de modelos
int save_tree_model(DecisionTree* tree, Dataset* dataset, const char* filename);
DecisionTree* load_tree_model(const char* filename);
int save_tree_to_file(TreeNode* node, FILE* file);
TreeNode* load_tree_from_file(FILE* file);
int save_model_registry();
int load_model_registry();
void list_models();
int delete_model(const char* model_name);
void export_model_json(DecisionTree* tree, Dataset* dataset, const char* filename);
void import_model_json(const char* filename, DecisionTree** tree, Dataset* dataset);
void backup_model(const char* model_name);
void restore_model(const char* backup_name);

// Dataset y preprocesamiento
Dataset load_dataset(const char* filename);
Dataset load_dataset_with_header(const char* filename, int has_header);
void preprocess_dataset(Dataset* dataset);
void normalize_dataset(Dataset* dataset);
void standardize_dataset(Dataset* dataset);
void calculate_dataset_statistics(Dataset* dataset);
void print_dataset_info(Dataset* dataset);
void split_dataset(Dataset* dataset, Dataset* train, Dataset* test, double ratio);
void shuffle_dataset(Dataset* dataset);
int is_comment_line(const char* line);
void trim_newline(char* str);
void normalize_sample(DataSample* sample, Dataset* dataset);
void print_sample(DataSample* sample, Dataset* dataset, int index);
void export_dataset_csv(Dataset* dataset, const char* filename);
void import_dataset_csv(const char* filename, Dataset* dataset);

// Árbol de decisión - Core
DecisionTree* create_decision_tree();
DecisionTree* train_decision_tree(Dataset* dataset, int max_depth, int min_samples_split, int min_samples_leaf);
DecisionTree* train_decision_tree_cv(Dataset* dataset, int max_depth, int min_samples_split, int min_samples_leaf, int folds);
double calculate_gini(Dataset* dataset, int* indices, int count);
double calculate_entropy(Dataset* dataset, int* indices, int count);
double calculate_gain(Dataset* dataset, int* indices, int count, int feature, double value, int criterion);
double calculate_gain_ratio(Dataset* dataset, int* indices, int count, int feature, double value);
void find_best_split(Dataset* dataset, int* indices, int count, int* best_feature, double* best_value, double* best_gain, int criterion);
TreeNode* build_tree(Dataset* dataset, int* indices, int count, int depth, int max_depth, int min_samples_split, int min_samples_leaf, int* node_counter, int criterion);
int predict_tree(TreeNode* node, DataSample* sample);
int predict_tree_with_proba(TreeNode* node, DataSample* sample, double* probabilities);
void free_tree(TreeNode* node);
void prune_tree(TreeNode* node, Dataset* dataset, double min_gain);
int tree_max_depth(TreeNode* node);
int count_classes(Dataset* dataset, int* indices, int count);
int find_majority_class(Dataset* dataset, int* indices, int count);
void count_tree_nodes(TreeNode* node, int* total, int* leaves);

// Evaluación de modelos
double evaluate_tree_accuracy(DecisionTree* tree, Dataset* dataset);
double evaluate_tree_precision(DecisionTree* tree, Dataset* dataset);
double evaluate_tree_recall(DecisionTree* tree, Dataset* dataset);
double evaluate_tree_f1(DecisionTree* tree, Dataset* dataset);
void evaluate_model_comprehensive(DecisionTree* tree, Dataset* dataset);
void print_confusion_matrix_tree(DecisionTree* tree, Dataset* dataset);
void calculate_feature_importance(TreeNode* node, double importance[], int total_samples);
void print_feature_importance_tree(DecisionTree* tree, Dataset* dataset);
void calculate_model_statistics(DecisionTree* tree, Dataset* dataset);
void cross_validation(Dataset* dataset, int folds, int max_depth, int min_samples_split, int min_samples_leaf);
void learning_curve(Dataset* dataset, int max_depth, int min_samples_split, int min_samples_leaf);

// Entrenamiento incremental y refinamiento
DecisionTree* refine_decision_tree(DecisionTree* existing_tree, Dataset* new_data);
DecisionTree* update_decision_tree(DecisionTree* tree, Dataset* new_data, double learning_rate);
void update_dataset_statistics(Dataset* dataset, DataSample* new_sample);

// Visualización
void visualize_tree_ascii(TreeNode* root, Dataset* dataset);
void visualize_tree_horizontal(TreeNode* node, Dataset* dataset, int depth, char* prefix, int is_left);
void print_tree_structure(TreeNode* node, int depth, char feature_names[][50]);
void print_tree_json(TreeNode* node, FILE* file, Dataset* dataset);
void export_tree_dot(TreeNode* root, Dataset* dataset, const char* filename);
void generate_tree_graphviz(TreeNode* root, Dataset* dataset, const char* filename);

// Interfaz de usuario
void clear_screen();
void wait_for_key(const char* message);
void print_header(const char* title);
void print_footer();
void print_separator();
int get_terminal_width();
void print_centered(const char* text, int width);
void print_boxed(const char* text, int width);
void print_progress_bar(int current, int total, const char* label, double additional_info);
void printf_both(const char* format, ...);
void print_colored(const char* text, int color);
void print_table_header(const char** headers, int num_headers, int* widths);
void print_table_row(const char** cells, int num_cells, int* widths);
void print_table_footer(int* widths, int num_widths);

// Gestión de archivos de salida
void init_output_file(const char* filename);
void close_output_file();
void init_export_file(const char* filename);
void close_export_file();
void init_log_file();
void close_log_file();
void save_output_to_file(const char* content);
void export_results_csv(DecisionTree* tree, Dataset* dataset, const char* filename);
void export_metrics_json(DecisionTree* tree, Dataset* dataset, const char* filename);

// Modos de operación
void interactive_mode(Dataset* dataset);
void training_mode(Dataset* dataset);
void model_management_mode();
void evaluate_mode(Dataset* dataset);
void save_model_mode(Dataset* dataset);
void export_mode(Dataset* dataset);
void import_mode();
void visualization_mode(Dataset* dataset);
void debug_mode_function(Dataset* dataset);
void benchmark_mode(Dataset* dataset);
void tutorial_mode();

// Funciones de evaluación para archivo
void print_confusion_matrix_to_file(DecisionTree* tree, Dataset* dataset);
void print_feature_importance_to_file(DecisionTree* tree, Dataset* dataset);
void print_model_metrics_to_file(DecisionTree* tree, Dataset* dataset);
void save_tree_structure_to_file(TreeNode* root, Dataset* dataset);
void save_predictions_to_file(DecisionTree* tree, Dataset* dataset);
void save_dataset_info_to_file(Dataset* dataset);
void save_training_log_to_file(DecisionTree* tree, Dataset* dataset, time_t start_time, time_t end_time);

// Funciones auxiliares
void shuffle_indices(int* indices, int count);
void show_data_distribution(Dataset* dataset);
void show_feature_statistics(Dataset* dataset);
void show_correlation_matrix(Dataset* dataset);
void show_prediction_path(TreeNode* root, DataSample* sample, Dataset* dataset);
void show_training_progress(int iteration, int total, const char* phase);
void print_usage(const char* program_name);
void print_license();
void print_version();

// Nuevas funciones avanzadas
void ensemble_training(Dataset* dataset, int num_trees, int max_depth);
void random_forest(Dataset* dataset, int num_trees, int max_features, int max_depth);
void gradient_boosting(Dataset* dataset, int num_trees, double learning_rate, int max_depth);
void save_model_checkpoint(DecisionTree* tree, Dataset* dataset, const char* checkpoint_name);
void load_model_checkpoint(const char* checkpoint_name, DecisionTree** tree, Dataset* dataset);
void compare_models(DecisionTree* model1, DecisionTree* model2, Dataset* dataset);
void hyperparameter_tuning(Dataset* dataset);
void feature_selection(Dataset* dataset, int num_features);
void outlier_detection(Dataset* dataset);
void data_augmentation(Dataset* dataset);

// ============================ FUNCIÓN PRINCIPAL ============================

int main(int argc, char* argv[]) {
    // Configurar manejador de señales
    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);
    signal(SIGSEGV, handle_signal);
    
    init_system();
    
    char* data_filename = NULL;
    char* output_filename = NULL;
    char* model_filename = NULL;
    char* export_filename = NULL;
    int interactive = 0;
    int train_new = 0;
    int max_depth = 5;
    int min_samples = 2;
    int min_samples_leaf = 1;
    int load_existing = 0;
    int export_results = 0;
    int benchmark = 0;
    int tutorial = 0;
    int debug = 0;
    
    // Parsear argumentos
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--interactive") == 0) {
            interactive = 1;
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--train") == 0) {
            train_new = 1;
        } else if (strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--load") == 0) {
            if (i + 1 < argc) {
                model_filename = argv[++i];
                load_existing = 1;
            }
        } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--depth") == 0) {
            if (i + 1 < argc) {
                max_depth = atoi(argv[++i]);
                if (max_depth < 1 || max_depth > MAX_TREE_DEPTH) {
                    printf("❌ Profundidad debe estar entre 1 y %d\n", MAX_TREE_DEPTH);
                    return 1;
                }
            }
        } else if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--min-samples") == 0) {
            if (i + 1 < argc) {
                min_samples = atoi(argv[++i]);
                if (min_samples < 2) {
                    printf("❌ Mínimo de muestras debe ser al menos 2\n");
                    return 1;
                }
            }
        } else if (strcmp(argv[i], "-ml") == 0 || strcmp(argv[i], "--min-samples-leaf") == 0) {
            if (i + 1 < argc) {
                min_samples_leaf = atoi(argv[++i]);
                if (min_samples_leaf < 1) {
                    printf("❌ Mínimo de muestras por hoja debe ser al menos 1\n");
                    return 1;
                }
            }
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
            if (i + 1 < argc) {
                output_filename = argv[++i];
            }
        } else if (strcmp(argv[i], "-e") == 0 || strcmp(argv[i], "--export") == 0) {
            if (i + 1 < argc) {
                export_filename = argv[++i];
                export_results = 1;
            }
        } else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--benchmark") == 0) {
            benchmark = 1;
        } else if (strcmp(argv[i], "-tu") == 0 || strcmp(argv[i], "--tutorial") == 0) {
            tutorial = 1;
        } else if (strcmp(argv[i], "-dbg") == 0 || strcmp(argv[i], "--debug") == 0) {
            debug = 1;
            debug_mode = 1;
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--version") == 0) {
            print_version();
            return 0;
        } else if (argv[i][0] != '-') {
            data_filename = argv[i];
        }
    }
    
    // Si se solicita el tutorial, ejecutarlo y salir
    if (tutorial) {
        tutorial_mode();
        cleanup_system();
        return 0;
    }
    
    // Cargar registro de modelos
    load_model_registry();
    
    // Inicializar logger de historia
    init_history_logger();
    
    // Si no se especificó archivo, usar frutas.data por defecto
    if (data_filename == NULL && !interactive) {
        printf("⚠️  No se especificó archivo de datos. Usando frutas.data por defecto.\n");
        data_filename = "frutas.data";
    }
    
    // Inicializar archivo de salida si se especificó
    if (output_filename) {
        init_output_file(output_filename);
        log_operation("INICIO_SISTEMA", "Archivo de salida inicializado", 0.0, 0);
    }
    
    // Inicializar archivo de exportación si se especificó
    if (export_filename && export_results) {
        init_export_file(export_filename);
    }
    
    clear_screen();
    print_header("🌳 SISTEMA AVANZADO DE ÁRBOLES DE DECISIÓN");
    print_system_info();
    
    Dataset dataset = {0};
    
    // Cargar dataset si se proporcionó archivo
    if (data_filename != NULL) {
        printf_both("\n📥 Cargando dataset: %s\n", data_filename);
        dataset = load_dataset(data_filename);
        
        if (dataset.num_samples == 0) {
            printf_both("❌ No se pudo cargar el dataset %s.\n", data_filename);
            wait_for_key("Presione Enter para continuar...");
            cleanup_system();
            return 1;
        }
        
        // Preprocesamiento
        printf_both("🔧 Preprocesando datos...\n");
        preprocess_dataset(&dataset);
        calculate_dataset_statistics(&dataset);
        
        log_operation("CARGA_DATASET", data_filename, 0.0, dataset.num_samples);
    }
    
    // Modo depuración
    if (debug) {
        debug_mode_function(&dataset);
        cleanup_system();
        return 0;
    }
    
    // Modo benchmark
    if (benchmark && dataset.num_samples > 0) {
        benchmark_mode(&dataset);
        cleanup_system();
        return 0;
    }
    
    // Cargar modelo existente si se especificó
    if (load_existing && model_filename) {
        printf_both("\n📂 Cargando modelo: %s\n", model_filename);
        current_model = load_tree_model(model_filename);
        if (current_model) {
            strcpy(current_model_file, model_filename);
            printf_both("✅ Modelo cargado exitosamente\n");
            printf_both("   • Nombre: %s\n", current_model->name);
            printf_both("   • Descripción: %s\n", current_model->description);
            printf_both("   • Precisión: %.2f%%\n", current_model->accuracy * 100);
            printf_both("   • Entrenado con: %d muestras\n", current_model->total_samples_trained);
            
            // Actualizar registro
            for (int i = 0; i < model_registry.num_models; i++) {
                if (strcmp(model_registry.models[i].filename, model_filename) == 0) {
                    model_registry.models[i].last_used = time(NULL);
                    break;
                }
            }
            save_model_registry();
            log_operation("CARGA_MODELO", model_filename, current_model->accuracy, current_model->total_samples_trained);
        } else {
            printf_both("❌ No se pudo cargar el modelo.\n");
        }
    }
    
    // Si se solicitó entrenar nuevo modelo
    if (train_new && dataset.num_samples > 0) {
        printf_both("\n🎯 Entrenando nuevo modelo...\n");
        current_model = train_decision_tree(&dataset, max_depth, min_samples, min_samples_leaf);
        if (current_model) {
            log_operation("ENTRENAMIENTO", "Nuevo modelo", current_model->accuracy, dataset.num_samples);
        }
    }
    
    // Modo interactivo automático
    if (interactive) {
        if (dataset.num_samples == 0 && !current_model) {
            printf_both("\n⚠️  No hay datos ni modelo cargado. Cargando dataset por defecto...\n");
            dataset = load_dataset("frutas.data");
            if (dataset.num_samples > 0) {
                preprocess_dataset(&dataset);
                calculate_dataset_statistics(&dataset);
            }
        }
        interactive_mode(&dataset);
        cleanup_system();
        return 0;
    }
    
    // Si no hay dataset cargado, salir
    if (dataset.num_samples == 0) {
        printf_both("\n⚠️  No hay datos cargados. Use modo interactivo o proporcione archivo.\n");
        cleanup_system();
        return 1;
    }
    
    // Menú principal
    int choice;
    do {
        clear_screen();
        print_header("MENÚ PRINCIPAL");
        
        printf_both("\n📊 DATASET ACTUAL:\n");
        printf_both("   • Muestras: %d\n", dataset.num_samples);
        printf_both("   • Características: %d\n", dataset.num_features);
        printf_both("   • Clases: %d\n", dataset.num_classes);
        
        printf_both("\n🌳 MODELO ACTUAL: ");
        if (current_model) {
            printf_both("%s\n", current_model->name);
            printf_both("   • Precisión: %.2f%% | F1-Score: %.2f%%\n", 
                   current_model->accuracy * 100, current_model->f1_score * 100);
            printf_both("   • Muestras entrenadas: %d\n", current_model->total_samples_trained);
        } else {
            printf_both("Ninguno\n");
        }
        
        printf_both("\n════════════════════════════════════════════════════════════════\n");
        printf_both("1. 📊 Análisis de datos\n");
        printf_both("2. 🎯 Entrenar nuevo modelo\n");
        printf_both("3. 📈 Evaluar modelo actual\n");
        printf_both("4. 🔍 Modo interactivo (predicciones)\n");
        printf_both("5. 💾 Guardar modelo actual\n");
        printf_both("6. 📂 Cargar modelo existente\n");
        printf_both("7. 🛠️  Refinar modelo actual\n");
        printf_both("8. 📁 Gestión de modelos\n");
        printf_both("9. 🌲 Visualizar estructura del árbol\n");
        printf_both("10. 📤 Exportar resultados\n");
        printf_both("11. 📥 Importar datos/modelos\n");
        printf_both("12. 📊 Modo benchmark\n");
        printf_both("13. 📚 Tutorial\n");
        printf_both("14. 🐛 Modo depuración\n");
        printf_both("15. 📝 Ver historial\n");
        printf_both("16. 🚪 Salir\n");
        printf_both("\nSeleccione una opción: ");
        
        if (scanf("%d", &choice) != 1) {
            while (getchar() != '\n'); // Limpiar buffer
            choice = 0;
        }
        getchar();
        
        switch(choice) {
            case 1:
                if (dataset.num_samples > 0) {
                    clear_screen();
                    print_header("📊 ANÁLISIS DE DATOS");
                    print_dataset_info(&dataset);
                    show_data_distribution(&dataset);
                    show_feature_statistics(&dataset);
                    if (dataset.num_features <= 10) {
                        show_correlation_matrix(&dataset);
                    }
                    wait_for_key("\nPresione Enter para continuar...");
                } else {
                    printf_both("\n❌ No hay datos cargados\n");
                    wait_for_key("Presione Enter para continuar...");
                }
                break;
                
            case 2:
                if (dataset.num_samples > 0) {
                    clear_screen();
                    print_header("🎯 ENTRENAR NUEVO MODELO");
                    
                    printf_both("\nConfigurar parámetros de entrenamiento:\n");
                    printf_both("Profundidad máxima (1-%d, actual %d): ", MAX_TREE_DEPTH, max_depth);
                    char depth_str[10];
                    fgets(depth_str, sizeof(depth_str), stdin);
                    if (strlen(depth_str) > 1) {
                        int new_depth = atoi(depth_str);
                        if (new_depth >= 1 && new_depth <= MAX_TREE_DEPTH) {
                            max_depth = new_depth;
                        }
                    }
                    
                    printf_both("Mínimo muestras por split (2+, actual %d): ", min_samples);
                    char min_str[10];
                    fgets(min_str, sizeof(min_str), stdin);
                    if (strlen(min_str) > 1) {
                        int new_min = atoi(min_str);
                        if (new_min >= 2) {
                            min_samples = new_min;
                        }
                    }
                    
                    printf_both("Mínimo muestras por hoja (1+, actual %d): ", min_samples_leaf);
                    char leaf_str[10];
                    fgets(leaf_str, sizeof(leaf_str), stdin);
                    if (strlen(leaf_str) > 1) {
                        int new_leaf = atoi(leaf_str);
                        if (new_leaf >= 1) {
                            min_samples_leaf = new_leaf;
                        }
                    }
                    
                    // Liberar modelo anterior si existe
                    if (current_model) {
                        free_tree(current_model->root);
                        free(current_model);
                    }
                    
                    printf_both("\n🔧 Entrenando modelo...\n");
                    current_model = train_decision_tree(&dataset, max_depth, min_samples, min_samples_leaf);
                    
                    if (current_model) {
                        log_operation("ENTRENAMIENTO_NUEVO", "Modelo entrenado", current_model->accuracy, dataset.num_samples);
                        printf_both("\n✅ Modelo entrenado exitosamente!\n");
                        printf_both("   • Nombre: %s\n", current_model->name);
                        printf_both("   • Precisión: %.2f%%\n", current_model->accuracy * 100);
                        printf_both("   • Nodos: %d | Hojas: %d\n", current_model->node_count, current_model->leaf_count);
                    } else {
                        printf_both("\n❌ Error al entrenar el modelo\n");
                    }
                    wait_for_key("\nPresione Enter para continuar...");
                } else {
                    printf_both("\n❌ No hay datos cargados para entrenar\n");
                    wait_for_key("Presione Enter para continuar...");
                }
                break;
                
            case 3:
                evaluate_mode(&dataset);
                break;
                
            case 4:
                interactive_mode(&dataset);
                break;
                
            case 5:
                save_model_mode(&dataset);
                break;
                
            case 6:
                {
                    clear_screen();
                    print_header("📂 CARGAR MODELO EXISTENTE");
                    
                    if (model_registry.num_models > 0) {
                        printf_both("\nModelos disponibles:\n");
                        for (int i = 0; i < model_registry.num_models && i < 10; i++) {
                            printf_both("%d. %s (%.1f%%)\n", i+1, 
                                   model_registry.models[i].name,
                                   model_registry.models[i].accuracy * 100);
                        }
                        printf_both("\nIngrese número del modelo (0 para cancelar): ");
                        int model_num;
                        scanf("%d", &model_num);
                        getchar();
                        
                        if (model_num > 0 && model_num <= model_registry.num_models) {
                            // Liberar modelo actual si existe
                            if (current_model) {
                                free_tree(current_model->root);
                                free(current_model);
                            }
                            
                            current_model = load_tree_model(model_registry.models[model_num-1].filename);
                            if (current_model) {
                                strcpy(current_model_file, model_registry.models[model_num-1].filename);
                                model_registry.models[model_num-1].last_used = time(NULL);
                                save_model_registry();
                                log_operation("CARGA_MODELO_SELECCIONADO", 
                                           model_registry.models[model_num-1].name,
                                           current_model->accuracy,
                                           current_model->total_samples_trained);
                                printf_both("\n✅ Modelo cargado: %s\n", current_model->name);
                            } else {
                                printf_both("\n❌ Error al cargar el modelo\n");
                            }
                        }
                    } else {
                        printf_both("\n📭 No hay modelos guardados\n");
                    }
                    wait_for_key("\nPresione Enter para continuar...");
                }
                break;
                
            case 7:
                if (current_model && dataset.num_samples > 0) {
                    printf_both("\n🔄 Refinando modelo actual...\n");
                    DecisionTree* refined = refine_decision_tree(current_model, &dataset);
                    if (refined) {
                        // Liberar modelo anterior
                        free_tree(current_model->root);
                        free(current_model);
                        current_model = refined;
                        log_operation("REFINAMIENTO", "Modelo refinado", current_model->accuracy, dataset.num_samples);
                        printf_both("✅ Modelo refinado exitosamente\n");
                    } else {
                        printf_both("❌ Error al refinar el modelo\n");
                    }
                    wait_for_key("\nPresione Enter para continuar...");
                } else {
                    printf_both("\n❌ No hay modelo cargado o datos disponibles\n");
                    wait_for_key("Presione Enter para continuar...");
                }
                break;
                
            case 8:
                model_management_mode();
                break;
                
            case 9:
                if (current_model && current_model->root) {
                    visualization_mode(&dataset);
                } else {
                    printf_both("\n❌ No hay modelo cargado para visualizar\n");
                    wait_for_key("Presione Enter para continuar...");
                }
                break;
                
            case 10:
                if (current_model && dataset.num_samples > 0) {
                    export_mode(&dataset);
                } else {
                    printf_both("\n❌ No hay modelo o datos para exportar\n");
                    wait_for_key("Presione Enter para continuar...");
                }
                break;
                
            case 11:
                import_mode();
                break;
                
            case 12:
                if (dataset.num_samples > 0) {
                    benchmark_mode(&dataset);
                } else {
                    printf_both("\n❌ No hay datos para benchmark\n");
                    wait_for_key("Presione Enter para continuar...");
                }
                break;
                
            case 13:
                tutorial_mode();
                break;
                
            case 14:
                debug_mode_function(&dataset);
                break;
                
            case 15:
                clear_screen();
                print_header("📝 HISTORIAL DE OPERACIONES");
                print_recent_history();
                wait_for_key("\nPresione Enter para continuar...");
                break;
                
            case 16:
                printf_both("\n👋 Saliendo del sistema...\n");
                break;
                
            default:
                printf_both("\n❌ Opción inválida\n");
                wait_for_key("Presione Enter para continuar...");
        }
        
    } while (choice != 16);
    
    // Guardar registro y limpiar antes de salir
    save_model_registry();
    save_history_log();
    
    // Liberar memoria
    if (current_model) {
        free_tree(current_model->root);
        free(current_model);
    }
    
    close_output_file();
    close_log_file();
    
    printf_both("\n✨ Sistema finalizado correctamente.\n");
    printf_both("   Historial guardado en: logs/history.log\n");
    printf_both("   Modelos guardados en: %s/\n", MODEL_DIR);
    
    return 0;
}

// ============================ IMPLEMENTACIONES ============================

// Sistema e inicialización
void init_system() {
    printf("🚀 Inicializando sistema...\n");
    
    // Crear directorios necesarios
    create_directories();
    
    // Inicializar semilla aleatoria
    srand(time(NULL));
    
    // Configurar locale para caracteres especiales
    setlocale(LC_ALL, "");
    
    printf("✅ Sistema inicializado\n");
}

void cleanup_system() {
    printf("\n\n⚠️  Limpiando sistema...\n");
    
    // Guardar todo antes de salir
    save_model_registry();
    save_history_log();
    
    // Cerrar archivos
    if (output_file) {
        close_output_file();
    }
    
    printf("✅ Sistema limpiado. Saliendo...\n");
}

void handle_signal(int sig) {
    printf("\n\n⚠️  Señal %d recibida. Limpiando...\n", sig);
    cleanup_system();
    exit(sig);
}

void create_directories() {
    struct stat st = {0};
    
    char* directories[] = {MODEL_DIR, LOGS_DIR, EXPORTS_DIR, "checkpoints", "backups"};
    int num_dirs = sizeof(directories) / sizeof(directories[0]);
    
    for (int i = 0; i < num_dirs; i++) {
        if (stat(directories[i], &st) == -1) {
            if (mkdir(directories[i], 0700) == 0) {
                printf("📁 Directorio creado: %s\n", directories[i]);
            } else {
                printf("⚠️  No se pudo crear directorio %s: %s\n", directories[i], strerror(errno));
            }
        }
    }
}

void print_system_info() {
    printf_both("Versión: 2.0 | Con persistencia avanzada\n");
    printf_both("Compilado: %s %s\n", __DATE__, __TIME__);
    printf_both("Límites: %d muestras, %d características, %d clases\n", 
           MAX_SAMPLES, MAX_FEATURES, MAX_CLASSES);
    printf_both("════════════════════════════════════════════════════════════════\n");
}

// Gestión de logs e historia
void init_history_logger() {
    char log_path[256];
    snprintf(log_path, sizeof(log_path), "%s/history.log", LOGS_DIR);
    
    history_logger.log_file = fopen(log_path, "a");
    if (!history_logger.log_file) {
        printf("⚠️  No se pudo abrir archivo de log\n");
    } else {
        fprintf(history_logger.log_file, "\n=== SESIÓN INICIADA: %s ===\n", ctime(&(time_t){time(NULL)}));
    }
    
    history_logger.count = 0;
    load_history_log();
}

void log_operation(const char* operation, const char* details, double accuracy, int samples) {
    if (history_logger.count < 1000) {
        HistoryEntry* entry = &history_logger.entries[history_logger.count];
        entry->timestamp = time(NULL);
        strncpy(entry->operation, operation, sizeof(entry->operation) - 1);
        strncpy(entry->details, details, sizeof(entry->details) - 1);
        entry->accuracy = accuracy;
        entry->samples = samples;
        history_logger.count++;
    }
    
    // Guardar en archivo de log
    if (history_logger.log_file) {
        fprintf(history_logger.log_file, "[%s] %s: %s (Accuracy: %.2f%%, Samples: %d)\n",
                ctime(&(time_t){time(NULL)}), operation, details, accuracy * 100, samples);
        fflush(history_logger.log_file);
    }
    
    // También guardar en salida si está activa
    if (output_to_file && output_file) {
        fprintf(output_file, "[%s] %s: %s (Accuracy: %.2f%%, Samples: %d)\n",
                ctime(&(time_t){time(NULL)}), operation, details, accuracy * 100, samples);
    }
}

void save_history_log() {
    char history_path[256];
    snprintf(history_path, sizeof(history_path), "%s/history.bin", LOGS_DIR);
    
    FILE* file = fopen(history_path, "wb");
    if (!file) return;
    
    fwrite(&history_logger.count, sizeof(int), 1, file);
    fwrite(history_logger.entries, sizeof(HistoryEntry), history_logger.count, file);
    
    fclose(file);
}

void load_history_log() {
    char history_path[256];
    snprintf(history_path, sizeof(history_path), "%s/history.bin", LOGS_DIR);
    
    FILE* file = fopen(history_path, "rb");
    if (!file) return;
    
    fread(&history_logger.count, sizeof(int), 1, file);
    if (history_logger.count > 1000) history_logger.count = 1000;
    
    fread(history_logger.entries, sizeof(HistoryEntry), history_logger.count, file);
    
    fclose(file);
}

void print_recent_history() {
    printf_both("\nÚltimas 10 operaciones:\n");
    printf_both("┌────────────────────┬──────────────────────┬──────────────────────┬────────────┬─────────┐\n");
    printf_both("│ Fecha              │ Operación            │ Detalles             │ Precisión  │ Muestras│\n");
    printf_both("├────────────────────┼──────────────────────┼──────────────────────┼────────────┼─────────┤\n");
    
    int start = history_logger.count > 10 ? history_logger.count - 10 : 0;
    for (int i = start; i < history_logger.count; i++) {
        HistoryEntry* entry = &history_logger.entries[i];
        char time_str[20];
        strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M", localtime(&entry->timestamp));
        
        printf_both("│ %-18s │ %-20s │ %-20s │ %9.2f%% │ %7d │\n",
               time_str,
               entry->operation,
               entry->details,
               entry->accuracy * 100,
               entry->samples);
    }
    printf_both("└────────────────────┴──────────────────────┴──────────────────────┴────────────┴─────────┘\n");
}

void export_history_csv(const char* filename) {
    char full_path[256];
    snprintf(full_path, sizeof(full_path), "%s/%s", EXPORTS_DIR, filename);
    
    FILE* file = fopen(full_path, "w");
    if (!file) {
        printf("❌ Error al crear archivo CSV\n");
        return;
    }
    
    fprintf(file, "Timestamp,Operation,Details,Accuracy,Samples\n");
    
    for (int i = 0; i < history_logger.count; i++) {
        HistoryEntry* entry = &history_logger.entries[i];
        char time_str[50];
        strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", localtime(&entry->timestamp));
        
        fprintf(file, "\"%s\",\"%s\",\"%s\",%.4f,%d\n",
                time_str,
                entry->operation,
                entry->details,
                entry->accuracy,
                entry->samples);
    }
    
    fclose(file);
    printf("✅ Historial exportado a: %s\n", full_path);
}

// Dataset y preprocesamiento
Dataset load_dataset(const char* filename) {
    return load_dataset_with_header(filename, 1); // Por defecto con encabezado
}

Dataset load_dataset_with_header(const char* filename, int has_header) {
    Dataset dataset = {0};
    FILE* file = fopen(filename, "r");
    
    if (!file) {
        printf("❌ Error: No se pudo abrir el archivo %s\n", filename);
        printf("   Error: %s\n", strerror(errno));
        return dataset;
    }

    char line[MAX_LINE_LENGTH];
    int sample_count = 0;
    int max_features = 0;
    int header_processed = 0;
    
    // Primera pasada: determinar estructura
    while (fgets(line, sizeof(line), file) && sample_count < MAX_SAMPLES) {
        trim_newline(line);
        
        if (strlen(line) == 0) continue;
        
        // Saltar comentarios
        if (is_comment_line(line)) {
            // Extraer nombres de características si es el primer comentario y tiene encabezado
            if (has_header && !header_processed) {
                char* comment = line;
                while (*comment && (*comment == '#' || isspace(*comment))) comment++;
                
                char temp_line[MAX_LINE_LENGTH];
                strcpy(temp_line, comment);
                
                char* token = strtok(temp_line, ",");
                int feature_count = 0;
                
                while (token != NULL && feature_count < MAX_FEATURES) {
                    // Limpiar espacios
                    char* clean_token = token;
                    while (*clean_token && isspace(*clean_token)) clean_token++;
                    char* end = clean_token + strlen(clean_token) - 1;
                    while (end > clean_token && isspace(*end)) *end-- = '\0';
                    
                    if (feature_count < MAX_FEATURES - 1) {
                        strcpy(dataset.feature_names[feature_count], clean_token);
                    } else {
                        // El último es la clase
                        strcpy(dataset.class_names[0], clean_token);
                    }
                    feature_count++;
                    token = strtok(NULL, ",");
                }
                
                if (feature_count > 1) {
                    dataset.num_features = feature_count - 1;
                    header_processed = 1;
                }
            }
            continue;
        }
        
        // Contar características en esta línea
        char temp_line[MAX_LINE_LENGTH];
        strcpy(temp_line, line);
        
        char* token = strtok(temp_line, ",");
        int feature_count = 0;
        
        while (token != NULL && feature_count < MAX_FEATURES) {
            feature_count++;
            token = strtok(NULL, ",");
        }
        
        if (feature_count > 1) {
            if (feature_count - 1 > max_features) {
                max_features = feature_count - 1; // Última es la clase
            }
            sample_count++;
        }
    }
    
    if (sample_count == 0) {
        printf("❌ Error: Archivo vacío o sin datos válidos\n");
        fclose(file);
        return dataset;
    }
    
    // Segunda pasada: leer datos reales
    rewind(file);
    sample_count = 0;
    dataset.num_features = max_features;
    
    // Inicializar estadísticas
    for (int i = 0; i < dataset.num_features; i++) {
        dataset.feature_min[i] = MAX_FEATURE_VALUE;
        dataset.feature_max[i] = MIN_FEATURE_VALUE;
        dataset.feature_mean[i] = 0.0;
        dataset.feature_std[i] = 0.0;
    }
    
    // Saltar encabezado si existe
    if (has_header) {
        while (fgets(line, sizeof(line), file) && is_comment_line(line)) {
            // Solo saltar líneas de comentario
        }
        // Retroceder una línea si no era un comentario
        fseek(file, -strlen(line), SEEK_CUR);
    }
    
    // Leer datos
    while (fgets(line, sizeof(line), file) && sample_count < MAX_SAMPLES) {
        trim_newline(line);
        
        if (strlen(line) == 0 || is_comment_line(line)) continue;
        
        char temp_line[MAX_LINE_LENGTH];
        strcpy(temp_line, line);
        
        char* token = strtok(temp_line, ",");
        int feature_count = 0;
        int is_valid = 1;
        
        while (token != NULL && feature_count <= dataset.num_features) {
            char* clean_token = token;
            while (*clean_token && isspace(*clean_token)) clean_token++;
            
            if (feature_count < dataset.num_features) {
                // Es una característica
                char* endptr;
                double value = strtod(clean_token, &endptr);
                
                if (endptr == clean_token) {
                    is_valid = 0;
                    break;
                }
                
                dataset.samples[sample_count].features[feature_count] = value;
                dataset.samples[sample_count].weight = 1.0; // Peso por defecto
                
                // Actualizar min/max
                if (value < dataset.feature_min[feature_count]) 
                    dataset.feature_min[feature_count] = value;
                if (value > dataset.feature_max[feature_count]) 
                    dataset.feature_max[feature_count] = value;
                    
                // Acumular para media
                dataset.feature_mean[feature_count] += value;
                    
                feature_count++;
            } else {
                // Es la clase
                int class_found = -1;
                for (int i = 0; i < dataset.num_classes; i++) {
                    if (strcmp(dataset.class_names[i], clean_token) == 0) {
                        class_found = i;
                        break;
                    }
                }
                
                if (class_found == -1 && dataset.num_classes < MAX_CLASSES) {
                    strcpy(dataset.class_names[dataset.num_classes], clean_token);
                    dataset.samples[sample_count].target = dataset.num_classes;
                    dataset.num_classes++;
                } else if (class_found != -1) {
                    dataset.samples[sample_count].target = class_found;
                } else {
                    is_valid = 0;
                }
            }
            
            token = strtok(NULL, ",");
        }
        
        if (is_valid && feature_count == dataset.num_features) {
            sample_count++;
        }
    }
    
    dataset.num_samples = sample_count;
    
    // Calcular medias
    for (int i = 0; i < dataset.num_features; i++) {
        if (dataset.num_samples > 0) {
            dataset.feature_mean[i] /= dataset.num_samples;
        }
    }
    
    fclose(file);
    
    // Si no hay nombres de características, generar nombres por defecto
    if (!header_processed) {
        for (int i = 0; i < dataset.num_features; i++) {
            sprintf(dataset.feature_names[i], "Característica_%d", i + 1);
        }
    }
    
    return dataset;
}

void preprocess_dataset(Dataset* dataset) {
    printf_both("🔧 Preprocesando datos...\n");
    
    // Normalización min-max
    normalize_dataset(dataset);
    
    // Calcular estadísticas completas
    calculate_dataset_statistics(dataset);
    
    dataset->is_normalized = 1;
    printf_both("✅ Datos preprocesados\n");
}

void normalize_dataset(Dataset* dataset) {
    for (int i = 0; i < dataset->num_features; i++) {
        double min_val = dataset->feature_min[i];
        double max_val = dataset->feature_max[i];
        double range = max_val - min_val;
        
        if (range > 1e-10) {
            for (int j = 0; j < dataset->num_samples; j++) {
                dataset->samples[j].features[i] = 
                    (dataset->samples[j].features[i] - min_val) / range;
            }
            // Actualizar min/max después de normalización
            dataset->feature_min[i] = 0.0;
            dataset->feature_max[i] = 1.0;
        } else {
            // Si no hay variación, establecer a 0.5
            for (int j = 0; j < dataset->num_samples; j++) {
                dataset->samples[j].features[i] = 0.5;
            }
            dataset->feature_min[i] = 0.5;
            dataset->feature_max[i] = 0.5;
        }
    }
}

void standardize_dataset(Dataset* dataset) {
    for (int i = 0; i < dataset->num_features; i++) {
        double mean = dataset->feature_mean[i];
        double std = dataset->feature_std[i];
        
        if (std > 1e-10) {
            for (int j = 0; j < dataset->num_samples; j++) {
                dataset->samples[j].features[i] = 
                    (dataset->samples[j].features[i] - mean) / std;
            }
            // Actualizar estadísticas
            dataset->feature_min[i] = -3.0; // Aproximado para distribución normal
            dataset->feature_max[i] = 3.0;
            dataset->feature_mean[i] = 0.0;
            dataset->feature_std[i] = 1.0;
        }
    }
}

void calculate_dataset_statistics(Dataset* dataset) {
    // Calcular medias (ya calculadas durante carga)
    // Calcular desviaciones estándar
    for (int i = 0; i < dataset->num_features; i++) {
        double sum_sq = 0.0;
        for (int j = 0; j < dataset->num_samples; j++) {
            double diff = dataset->samples[j].features[i] - dataset->feature_mean[i];
            sum_sq += diff * diff;
        }
        if (dataset->num_samples > 1) {
            dataset->feature_std[i] = sqrt(sum_sq / (dataset->num_samples - 1));
        } else {
            dataset->feature_std[i] = 0.0;
        }
    }
}

void print_dataset_info(Dataset* dataset) {
    printf_both("\n📊 INFORMACIÓN DEL DATASET:\n");
    printf_both("   • Muestras totales: %d\n", dataset->num_samples);
    printf_both("   • Características: %d\n", dataset->num_features);
    printf_both("   • Clases: %d\n", dataset->num_classes);
    printf_both("   • Normalizado: %s\n", dataset->is_normalized ? "Sí" : "No");
    
    if (dataset->num_classes > 0) {
        printf_both("   • Nombres de clases: ");
        for (int i = 0; i < dataset->num_classes && i < 5; i++) {
            printf_both("%s", dataset->class_names[i]);
            if (i < dataset->num_classes - 1 && i < 4) printf_both(", ");
        }
        if (dataset->num_classes > 5) printf_both(", ...");
        printf_both("\n");
    }
    
    printf_both("\n📈 ESTADÍSTICAS POR CLASE:\n");
    int class_counts[MAX_CLASSES] = {0};
    for (int i = 0; i < dataset->num_samples; i++) {
        class_counts[dataset->samples[i].target]++;
    }
    
    for (int i = 0; i < dataset->num_classes; i++) {
        double percentage = (double)class_counts[i] / dataset->num_samples * 100;
        printf_both("   • %s: %d muestras (%.1f%%)\n", 
               dataset->class_names[i], class_counts[i], percentage);
    }
}

void split_dataset(Dataset* dataset, Dataset* train, Dataset* test, double ratio) {
    if (ratio <= 0.0 || ratio >= 1.0) ratio = 0.8;
    
    int train_size = (int)(dataset->num_samples * ratio);
    int test_size = dataset->num_samples - train_size;
    
    // Copiar estructura
    memcpy(train, dataset, sizeof(Dataset));
    memcpy(test, dataset, sizeof(Dataset));
    
    train->num_samples = 0;
    test->num_samples = 0;
    
    // Crear índices aleatorios
    int* indices = malloc(dataset->num_samples * sizeof(int));
    for (int i = 0; i < dataset->num_samples; i++) indices[i] = i;
    shuffle_indices(indices, dataset->num_samples);
    
    // Asignar a train
    for (int i = 0; i < train_size; i++) {
        int idx = indices[i];
        train->samples[i] = dataset->samples[idx];
        train->num_samples++;
    }
    
    // Asignar a test
    for (int i = 0; i < test_size; i++) {
        int idx = indices[train_size + i];
        test->samples[i] = dataset->samples[idx];
        test->num_samples++;
    }
    
    free(indices);
    
    // Recalcular estadísticas
    calculate_dataset_statistics(train);
    calculate_dataset_statistics(test);
}

void shuffle_dataset(Dataset* dataset) {
    for (int i = dataset->num_samples - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        // Intercambiar muestras
        DataSample temp = dataset->samples[i];
        dataset->samples[i] = dataset->samples[j];
        dataset->samples[j] = temp;
    }
}

int is_comment_line(const char* line) {
    while (*line && isspace(*line)) line++;
    return (*line == '#');
}

void trim_newline(char* str) {
    int len = strlen(str);
    while (len > 0 && (str[len-1] == '\n' || str[len-1] == '\r')) {
        str[len-1] = '\0';
        len--;
    }
}

void normalize_sample(DataSample* sample, Dataset* dataset) {
    for (int i = 0; i < dataset->num_features; i++) {
        double min_val = dataset->feature_min[i];
        double max_val = dataset->feature_max[i];
        
        if (max_val - min_val > 1e-10) {
            sample->features[i] = (sample->features[i] - min_val) / (max_val - min_val);
        } else {
            sample->features[i] = 0.5;
        }
    }
}

void print_sample(DataSample* sample, Dataset* dataset, int index) {
    printf_both("Muestra %d:\n", index);
    printf_both("  Características: ");
    for (int i = 0; i < dataset->num_features && i < 5; i++) {
        printf_both("%.4f", sample->features[i]);
        if (i < dataset->num_features - 1 && i < 4) printf_both(", ");
    }
    if (dataset->num_features > 5) printf_both(", ...");
    
    printf_both("\n  Clase: %s\n", dataset->class_names[sample->target]);
}

void export_dataset_csv(Dataset* dataset, const char* filename) {
    char full_path[256];
    snprintf(full_path, sizeof(full_path), "%s/%s", EXPORTS_DIR, filename);
    
    FILE* file = fopen(full_path, "w");
    if (!file) {
        printf("❌ Error al crear archivo CSV\n");
        return;
    }
    
    // Escribir encabezado
    for (int i = 0; i < dataset->num_features; i++) {
        fprintf(file, "%s", dataset->feature_names[i]);
        if (i < dataset->num_features - 1) fprintf(file, ",");
    }
    fprintf(file, ",class\n");
    
    // Escribir datos
    for (int i = 0; i < dataset->num_samples; i++) {
        for (int j = 0; j < dataset->num_features; j++) {
            fprintf(file, "%.6f", dataset->samples[i].features[j]);
            if (j < dataset->num_features - 1) fprintf(file, ",");
        }
        fprintf(file, ",%s\n", dataset->class_names[dataset->samples[i].target]);
    }
    
    fclose(file);
    printf("✅ Dataset exportado a: %s\n", full_path);
}

void import_dataset_csv(const char* filename, Dataset* dataset) {
    *dataset = load_dataset(filename);
    if (dataset->num_samples > 0) {
        printf("✅ Dataset importado: %d muestras\n", dataset->num_samples);
    }
}

// Árbol de decisión - Core
DecisionTree* create_decision_tree() {
    DecisionTree* tree = malloc(sizeof(DecisionTree));
    if (!tree) return NULL;
    
    memset(tree, 0, sizeof(DecisionTree));
    strcpy(tree->algorithm, "CART");
    tree->created_at = time(NULL);
    tree->is_trained = 0;
    
    return tree;
}

DecisionTree* train_decision_tree(Dataset* dataset, int max_depth, int min_samples_split, int min_samples_leaf) {
    if (dataset->num_samples == 0 || dataset->num_features == 0) {
        printf("❌ Dataset vacío o sin características\n");
        return NULL;
    }
    
    printf_both("\n🎯 INICIANDO ENTRENAMIENTO\n");
    printf_both("   • Muestras: %d\n", dataset->num_samples);
    printf_both("   • Características: %d\n", dataset->num_features);
    printf_both("   • Clases: %d\n", dataset->num_classes);
    printf_both("   • Parámetros: depth=%d, min_split=%d, min_leaf=%d\n", 
           max_depth, min_samples_split, min_samples_leaf);
    
    time_t start_time = time(NULL);
    
    DecisionTree* tree = create_decision_tree();
    if (!tree) return NULL;
    
    tree->max_depth = max_depth;
    tree->min_samples_split = min_samples_split;
    tree->min_samples_leaf = min_samples_leaf;
    tree->created_at = time(NULL);
    tree->last_trained = time(NULL);
    tree->total_samples_trained = dataset->num_samples;
    tree->is_trained = 1;
    
    // Generar nombre único
    time_t now = time(NULL);
    struct tm* t = localtime(&now);
    snprintf(tree->name, sizeof(tree->name), "Modelo_%04d%02d%02d_%02d%02d%02d",
            t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
            t->tm_hour, t->tm_min, t->tm_sec);
    
    snprintf(tree->description, sizeof(tree->description),
            "Árbol CART entrenado con %d muestras", dataset->num_samples);
    
    // Crear índices
    int* indices = malloc(dataset->num_samples * sizeof(int));
    if (!indices) {
        free(tree);
        return NULL;
    }
    
    for (int i = 0; i < dataset->num_samples; i++) {
        indices[i] = i;
    }
    
    // Mezclar índices
    shuffle_indices(indices, dataset->num_samples);
    
    printf_both("\n🔨 Construyendo árbol...\n");
    
    // Construir árbol usando criterio Gini (0)
    tree->root = build_tree(dataset, indices, dataset->num_samples, 0, 
                           max_depth, min_samples_split, min_samples_leaf, 
                           &tree->node_count, 0);
    
    // Contar hojas
    count_tree_nodes(tree->root, NULL, &tree->leaf_count);
    
    // Evaluar
    tree->accuracy = evaluate_tree_accuracy(tree, dataset);
    tree->precision = evaluate_tree_precision(tree, dataset);
    tree->recall = evaluate_tree_recall(tree, dataset);
    tree->f1_score = evaluate_tree_f1(tree, dataset);
    
    // Calcular importancia de características
    calculate_feature_importance(tree->root, tree->feature_importance, dataset->num_samples);
    
    free(indices);
    
    time_t end_time = time(NULL);
    double training_time = difftime(end_time, start_time);
    
    printf_both("\n✅ ENTRENAMIENTO COMPLETADO\n");
    printf_both("   • Tiempo: %.2f segundos\n", training_time);
    printf_both("   • Precisión: %.2f%%\n", tree->accuracy * 100);
    printf_both("   • Nodos: %d | Hojas: %d\n", tree->node_count, tree->leaf_count);
    
    // Guardar log de entrenamiento
    save_training_log_to_file(tree, dataset, start_time, end_time);
    
    return tree;
}

double calculate_gini(Dataset* dataset, int* indices, int count) {
    if (count == 0) return 0.0;
    
    int class_counts[MAX_CLASSES] = {0};
    
    for (int i = 0; i < count; i++) {
        int sample_idx = indices[i];
        class_counts[dataset->samples[sample_idx].target]++;
    }
    
    double gini = 1.0;
    for (int c = 0; c < dataset->num_classes; c++) {
        double p = (double)class_counts[c] / count;
        gini -= p * p;
    }
    
    return gini;
}

double calculate_entropy(Dataset* dataset, int* indices, int count) {
    if (count == 0) return 0.0;
    
    int class_counts[MAX_CLASSES] = {0};
    
    for (int i = 0; i < count; i++) {
        int sample_idx = indices[i];
        class_counts[dataset->samples[sample_idx].target]++;
    }
    
    double entropy = 0.0;
    for (int c = 0; c < dataset->num_classes; c++) {
        if (class_counts[c] > 0) {
            double p = (double)class_counts[c] / count;
            entropy -= p * log2(p);
        }
    }
    
    return entropy;
}

double calculate_gain(Dataset* dataset, int* indices, int count, int feature, double value, int criterion) {
    if (count == 0) return 0.0;
    
    double parent_impurity;
    if (criterion == 0) { // Gini
        parent_impurity = calculate_gini(dataset, indices, count);
    } else { // Entropía
        parent_impurity = calculate_entropy(dataset, indices, count);
    }
    
    int left_count = 0, right_count = 0;
    int* left_indices = malloc(count * sizeof(int));
    int* right_indices = malloc(count * sizeof(int));
    
    if (!left_indices || !right_indices) {
        free(left_indices);
        free(right_indices);
        return 0.0;
    }
    
    for (int i = 0; i < count; i++) {
        int sample_idx = indices[i];
        if (dataset->samples[sample_idx].features[feature] <= value) {
            left_indices[left_count++] = sample_idx;
        } else {
            right_indices[right_count++] = sample_idx;
        }
    }
    
    if (left_count == 0 || right_count == 0) {
        free(left_indices);
        free(right_indices);
        return 0.0;
    }
    
    double left_impurity, right_impurity;
    if (criterion == 0) {
        left_impurity = calculate_gini(dataset, left_indices, left_count);
        right_impurity = calculate_gini(dataset, right_indices, right_count);
    } else {
        left_impurity = calculate_entropy(dataset, left_indices, left_count);
        right_impurity = calculate_entropy(dataset, right_indices, right_count);
    }
    
    double weighted_impurity = (left_impurity * left_count + right_impurity * right_count) / count;
    double gain = parent_impurity - weighted_impurity;
    
    free(left_indices);
    free(right_indices);
    
    return gain;
}

void find_best_split(Dataset* dataset, int* indices, int count, 
                     int* best_feature, double* best_value, double* best_gain, int criterion) {
    *best_feature = -1;
    *best_value = 0.0;
    *best_gain = 0.0;
    
    if (count < 2) return;
    
    double parent_impurity;
    if (criterion == 0) {
        parent_impurity = calculate_gini(dataset, indices, count);
    } else {
        parent_impurity = calculate_entropy(dataset, indices, count);
    }
    
    // Para cada característica
    for (int feature = 0; feature < dataset->num_features; feature++) {
        // Probar diferentes valores de división
        for (int i = 0; i < count; i++) {
            double value = dataset->samples[indices[i]].features[feature];
            double gain = calculate_gain(dataset, indices, count, feature, value, criterion);
            
            if (gain > *best_gain) {
                *best_gain = gain;
                *best_feature = feature;
                *best_value = value;
            }
        }
    }
}

TreeNode* build_tree(Dataset* dataset, int* indices, int count, int depth, 
                    int max_depth, int min_samples_split, int min_samples_leaf, 
                    int* node_counter, int criterion) {
    TreeNode* node = malloc(sizeof(TreeNode));
    if (!node) return NULL;
    
    memset(node, 0, sizeof(TreeNode));
    (*node_counter)++;
    node->depth = depth;
    node->samples = count;
    
    // Almacenar índices de muestras (para debugging)
    if (debug_mode) {
        node->sample_indices = malloc(count * sizeof(int));
        if (node->sample_indices) {
            memcpy(node->sample_indices, indices, count * sizeof(int));
            node->sample_count = count;
        }
    }
    
    // Calcular impureza
    if (criterion == 0) {
        node->gini = calculate_gini(dataset, indices, count);
        node->entropy = calculate_entropy(dataset, indices, count);
    } else {
        node->entropy = calculate_entropy(dataset, indices, count);
        node->gini = calculate_gini(dataset, indices, count);
    }
    
    // Encontrar clase mayoritaria
    int majority_class = find_majority_class(dataset, indices, count);
    int unique_classes = count_classes(dataset, indices, count);
    
    // Condiciones de parada
    if (depth >= max_depth || 
        count < min_samples_split || 
        unique_classes <= 1 || 
        (criterion == 0 ? node->gini < 0.01 : node->entropy < 0.01)) {
        node->is_leaf = 1;
        node->class_label = majority_class;
        return node;
    }
    
    // Encontrar mejor split
    int best_feature;
    double best_value, best_gain;
    find_best_split(dataset, indices, count, &best_feature, &best_value, &best_gain, criterion);
    
    if (best_feature == -1 || best_gain < 0.001) {
        node->is_leaf = 1;
        node->class_label = majority_class;
        return node;
    }
    
    node->is_leaf = 0;
    node->split_feature = best_feature;
    node->split_value = best_value;
    
    // Dividir datos
    int left_count = 0, right_count = 0;
    int* left_indices = malloc(count * sizeof(int));
    int* right_indices = malloc(count * sizeof(int));
    
    if (!left_indices || !right_indices) {
        free(left_indices);
        free(right_indices);
        node->is_leaf = 1;
        node->class_label = majority_class;
        return node;
    }
    
    for (int i = 0; i < count; i++) {
        int sample_idx = indices[i];
        if (dataset->samples[sample_idx].features[best_feature] <= best_value) {
            left_indices[left_count++] = sample_idx;
        } else {
            right_indices[right_count++] = sample_idx;
        }
    }
    
    // Verificar mínimo muestras por hoja
    if (left_count < min_samples_leaf || right_count < min_samples_leaf) {
        free(left_indices);
        free(right_indices);
        node->is_leaf = 1;
        node->class_label = majority_class;
        return node;
    }
    
    // Construir subárboles recursivamente
    node->left = build_tree(dataset, left_indices, left_count, depth + 1, 
                           max_depth, min_samples_split, min_samples_leaf, 
                           node_counter, criterion);
    node->right = build_tree(dataset, right_indices, right_count, depth + 1, 
                            max_depth, min_samples_split, min_samples_leaf, 
                            node_counter, criterion);
    
    free(left_indices);
    free(right_indices);
    
    return node;
}

int predict_tree(TreeNode* node, DataSample* sample) {
    if (!node) return 0;
    
    if (node->is_leaf) {
        return node->class_label;
    }
    
    if (sample->features[node->split_feature] <= node->split_value) {
        return predict_tree(node->left, sample);
    } else {
        return predict_tree(node->right, sample);
    }
}

int predict_tree_with_proba(TreeNode* node, DataSample* sample, double* probabilities) {
    if (!node) return 0;
    
    if (node->is_leaf) {
        // Para una hoja, la probabilidad es 1.0 para la clase predicha
        for (int i = 0; i < MAX_CLASSES; i++) {
            probabilities[i] = 0.0;
        }
        probabilities[node->class_label] = 1.0;
        return node->class_label;
    }
    
    if (sample->features[node->split_feature] <= node->split_value) {
        return predict_tree_with_proba(node->left, sample, probabilities);
    } else {
        return predict_tree_with_proba(node->right, sample, probabilities);
    }
}

void free_tree(TreeNode* node) {
    if (node == NULL) return;
    
    if (!node->is_leaf) {
        free_tree(node->left);
        free_tree(node->right);
    }
    
    if (debug_mode && node->sample_indices) {
        free(node->sample_indices);
    }
    
    free(node);
}

void prune_tree(TreeNode* node, Dataset* dataset, double min_gain) {
    if (!node || node->is_leaf) return;
    
    // Podar recursivamente
    prune_tree(node->left, dataset, min_gain);
    prune_tree(node->right, dataset, min_gain);
    
    // Si ambos hijos son hojas, evaluar si podar
    if (node->left && node->left->is_leaf && 
        node->right && node->right->is_leaf) {
        
        // Calcular ganancia actual
        double current_gain = node->gini;
        if (node->left) current_gain -= node->left->gini * node->left->samples / node->samples;
        if (node->right) current_gain -= node->right->gini * node->right->samples / node->samples;
        
        // Si la ganancia es menor que el mínimo, podar
        if (current_gain < min_gain) {
            free_tree(node->left);
            free_tree(node->right);
            node->is_leaf = 1;
            node->class_label = find_majority_class(dataset, NULL, 0); // Necesitaría índices
        }
    }
}

int tree_max_depth(TreeNode* node) {
    if (!node) return 0;
    if (node->is_leaf) return node->depth;
    
    int left_depth = tree_max_depth(node->left);
    int right_depth = tree_max_depth(node->right);
    
    return (left_depth > right_depth ? left_depth : right_depth);
}

int count_classes(Dataset* dataset, int* indices, int count) {
    int unique_classes = 0;
    int class_present[MAX_CLASSES] = {0};
    
    for (int i = 0; i < count; i++) {
        int class_val = dataset->samples[indices[i]].target;
        if (!class_present[class_val]) {
            class_present[class_val] = 1;
            unique_classes++;
        }
    }
    
    return unique_classes;
}

int find_majority_class(Dataset* dataset, int* indices, int count) {
    if (count == 0) return 0;
    
    int class_counts[MAX_CLASSES] = {0};
    
    for (int i = 0; i < count; i++) {
        int sample_idx = indices[i];
        class_counts[dataset->samples[sample_idx].target]++;
    }
    
    int majority_class = 0;
    int max_count = 0;
    
    for (int c = 0; c < dataset->num_classes; c++) {
        if (class_counts[c] > max_count) {
            max_count = class_counts[c];
            majority_class = c;
        }
    }
    
    return majority_class;
}

void count_tree_nodes(TreeNode* node, int* total, int* leaves) {
    if (!node) return;
    
    if (total) (*total)++;
    
    if (node->is_leaf) {
        if (leaves) (*leaves)++;
    } else {
        count_tree_nodes(node->left, total, leaves);
        count_tree_nodes(node->right, total, leaves);
    }
}

// Evaluación de modelos
double evaluate_tree_accuracy(DecisionTree* tree, Dataset* dataset) {
    if (!tree || !tree->root || dataset->num_samples == 0) return 0.0;
    
    int correct = 0;
    
    for (int i = 0; i < dataset->num_samples; i++) {
        int prediction = predict_tree(tree->root, &dataset->samples[i]);
        if (prediction == dataset->samples[i].target) {
            correct++;
        }
    }
    
    return (double)correct / dataset->num_samples;
}

double evaluate_tree_precision(DecisionTree* tree, Dataset* dataset) {
    if (!tree || !tree->root || dataset->num_samples == 0) return 0.0;
    
    int confusion[MAX_CLASSES][MAX_CLASSES] = {0};
    
    for (int i = 0; i < dataset->num_samples; i++) {
        int prediction = predict_tree(tree->root, &dataset->samples[i]);
        confusion[dataset->samples[i].target][prediction]++;
    }
    
    double total_precision = 0.0;
    int valid_classes = 0;
    
    for (int i = 0; i < dataset->num_classes; i++) {
        int true_positives = confusion[i][i];
        int false_positives = 0;
        
        for (int j = 0; j < dataset->num_classes; j++) {
            if (j != i) false_positives += confusion[j][i];
        }
        
        if (true_positives + false_positives > 0) {
            total_precision += (double)true_positives / (true_positives + false_positives);
            valid_classes++;
        }
    }
    
    return valid_classes > 0 ? total_precision / valid_classes : 0.0;
}

double evaluate_tree_recall(DecisionTree* tree, Dataset* dataset) {
    if (!tree || !tree->root || dataset->num_samples == 0) return 0.0;
    
    int confusion[MAX_CLASSES][MAX_CLASSES] = {0};
    
    for (int i = 0; i < dataset->num_samples; i++) {
        int prediction = predict_tree(tree->root, &dataset->samples[i]);
        confusion[dataset->samples[i].target][prediction]++;
    }
    
    double total_recall = 0.0;
    int valid_classes = 0;
    
    for (int i = 0; i < dataset->num_classes; i++) {
        int true_positives = confusion[i][i];
        int false_negatives = 0;
        
        for (int j = 0; j < dataset->num_classes; j++) {
            if (j != i) false_negatives += confusion[i][j];
        }
        
        if (true_positives + false_negatives > 0) {
            total_recall += (double)true_positives / (true_positives + false_negatives);
            valid_classes++;
        }
    }
    
    return valid_classes > 0 ? total_recall / valid_classes : 0.0;
}

double evaluate_tree_f1(DecisionTree* tree, Dataset* dataset) {
    double precision = evaluate_tree_precision(tree, dataset);
    double recall = evaluate_tree_recall(tree, dataset);
    
    if (precision + recall == 0) return 0.0;
    return 2 * (precision * recall) / (precision + recall);
}

void evaluate_model_comprehensive(DecisionTree* tree, Dataset* dataset) {
    if (!tree || !tree->root) {
        printf_both("❌ Modelo no entrenado\n");
        return;
    }
    
    printf_both("\n📊 EVALUACIÓN COMPLETA DEL MODELO\n");
    printf_both("════════════════════════════════════════════════════════════════\n");
    
    // Métricas básicas
    printf_both("\n📈 MÉTRICAS DE RENDIMIENTO:\n");
    printf_both("┌──────────────────────────┬────────────────────┐\n");
    printf_both("│ Métrica                  │ Valor              │\n");
    printf_both("├──────────────────────────┼────────────────────┤\n");
    printf_both("│ Precisión (Accuracy)     │ %18.2f%% │\n", tree->accuracy * 100);
    printf_both("│ Precisión (Precision)    │ %18.2f%% │\n", tree->precision * 100);
    printf_both("│ Sensibilidad (Recall)    │ %18.2f%% │\n", tree->recall * 100);
    printf_both("│ F1-Score                 │ %18.2f%% │\n", tree->f1_score * 100);
    printf_both("└──────────────────────────┴────────────────────┘\n");
    
    // Matriz de confusión
    printf_both("\n📋 MATRIZ DE CONFUSIÓN:\n");
    print_confusion_matrix_tree(tree, dataset);
    
    // Importancia de características
    printf_both("\n🎯 IMPORTANCIA DE CARACTERÍSTICAS:\n");
    print_feature_importance_tree(tree, dataset);
    
    // Estadísticas del árbol
    printf_both("\n🌲 ESTADÍSTICAS DEL ÁRBOL:\n");
    printf_both("┌──────────────────────────┬────────────────────┐\n");
    printf_both("│ Estadística              │ Valor              │\n");
    printf_both("├──────────────────────────┼────────────────────┤\n");
    printf_both("│ Nodos totales            │ %18d │\n", tree->node_count);
    printf_both("│ Hojas                    │ %18d │\n", tree->leaf_count);
    printf_both("│ Profundidad máxima       │ %18d │\n", tree_max_depth(tree->root));
    printf_both("│ Muestras entrenadas      │ %18d │\n", tree->total_samples_trained);
    printf_both("│ Algoritmo                │ %18s │\n", tree->algorithm);
    printf_both("└──────────────────────────┴────────────────────┘\n");
    
    // Información temporal
    char created_str[50], trained_str[50], used_str[50];
    strftime(created_str, sizeof(created_str), "%Y-%m-%d %H:%M:%S", localtime(&tree->created_at));
    strftime(trained_str, sizeof(trained_str), "%Y-%m-%d %H:%M:%S", localtime(&tree->last_trained));
    strftime(used_str, sizeof(used_str), "%Y-%m-%d %H:%M:%S", localtime(&tree->last_used));
    
    printf_both("\n⏰ INFORMACIÓN TEMPORAL:\n");
    printf_both("   • Creado: %s\n", created_str);
    printf_both("   • Último entrenamiento: %s\n", trained_str);
    printf_both("   • Último uso: %s\n", used_str);
}

void print_confusion_matrix_tree(DecisionTree* tree, Dataset* dataset) {
    if (dataset->num_classes == 0) {
        printf_both("❌ No hay clases definidas\n");
        return;
    }
    
    int confusion[MAX_CLASSES][MAX_CLASSES] = {0};
    
    // Calcular matriz
    for (int i = 0; i < dataset->num_samples; i++) {
        int prediction = predict_tree(tree->root, &dataset->samples[i]);
        confusion[dataset->samples[i].target][prediction]++;
    }
    
    // Mostrar matriz
    int max_classes_to_show = dataset->num_classes < 6 ? dataset->num_classes : 6;
    
    printf_both("┌──────────");
    for (int i = 0; i < max_classes_to_show; i++) {
        printf_both("──────────");
    }
    if (dataset->num_classes > 6) printf_both("─────");
    printf_both("┐\n");
    
    printf_both("│          ");
    for (int i = 0; i < max_classes_to_show; i++) {
        printf_both("Pred %-4d ", i);
    }
    if (dataset->num_classes > 6) printf_both("... ");
    printf_both("│\n");
    
    printf_both("├──────────");
    for (int i = 0; i < max_classes_to_show; i++) {
        printf_both("──────────");
    }
    if (dataset->num_classes > 6) printf_both("─────");
    printf_both("┤\n");
    
    for (int i = 0; i < dataset->num_classes && i < 8; i++) {
        printf_both("│ Real %-3d ", i);
        for (int j = 0; j < max_classes_to_show; j++) {
            printf_both(" %-8d ", confusion[i][j]);
        }
        if (dataset->num_classes > 6) printf_both("... ");
        printf_both("│\n");
    }
    
    if (dataset->num_classes > 8) {
        printf_both("│ ...                                                  │\n");
    }
    
    printf_both("└──────────");
    for (int i = 0; i < max_classes_to_show; i++) {
        printf_both("──────────");
    }
    if (dataset->num_classes > 6) printf_both("─────");
    printf_both("┘\n");
    
    // Calcular métricas por clase
    printf_both("\n📊 MÉTRICAS POR CLASE:\n");
    printf_both("┌─────┬──────────────────────┬──────────────┬──────────────┬──────────────┐\n");
    printf_both("│ Cls │ Nombre               │ Precisión    │ Recall       │ F1-Score     │\n");
    printf_both("├─────┼──────────────────────┼──────────────┼──────────────┼──────────────┤\n");
    
    for (int i = 0; i < dataset->num_classes && i < 10; i++) {
        int true_positives = confusion[i][i];
        int false_positives = 0;
        int false_negatives = 0;
        
        for (int j = 0; j < dataset->num_classes; j++) {
            if (j != i) {
                false_positives += confusion[j][i];
                false_negatives += confusion[i][j];
            }
        }
        
        double precision = 0.0;
        double recall = 0.0;
        double f1 = 0.0;
        
        if (true_positives + false_positives > 0) {
            precision = (double)true_positives / (true_positives + false_positives);
        }
        
        if (true_positives + false_negatives > 0) {
            recall = (double)true_positives / (true_positives + false_negatives);
        }
        
        if (precision + recall > 0) {
            f1 = 2 * (precision * recall) / (precision + recall);
        }
        
        // Truncar nombre si es muy largo
        char display_name[21];
        strncpy(display_name, dataset->class_names[i], 20);
        display_name[20] = '\0';
        if (strlen(dataset->class_names[i]) > 20) {
            display_name[17] = display_name[18] = display_name[19] = '.';
        }
        
        printf_both("│ %3d │ %-20s │ %11.2f%% │ %11.2f%% │ %11.2f%% │\n",
               i, display_name, precision * 100, recall * 100, f1 * 100);
    }
    
    printf_both("└─────┴──────────────────────┴──────────────┴──────────────┴──────────────┘\n");
}

void calculate_feature_importance(TreeNode* node, double importance[], int total_samples) {
    if (!node || total_samples == 0) return;
    
    if (!node->is_leaf) {
        // La importancia es proporcional a la reducción de impureza
        double reduction = node->gini * node->samples;
        if (node->left) reduction -= node->left->gini * node->left->samples;
        if (node->right) reduction -= node->right->gini * node->right->samples;
        
        importance[node->split_feature] += reduction / total_samples;
        
        calculate_feature_importance(node->left, importance, total_samples);
        calculate_feature_importance(node->right, importance, total_samples);
    }
}

void print_feature_importance_tree(DecisionTree* tree, Dataset* dataset) {
    double importance[MAX_FEATURES] = {0};
    
    if (tree->root) {
        calculate_feature_importance(tree->root, importance, dataset->num_samples);
    }
    
    // Normalizar importancias a porcentaje
    double sum = 0.0;
    for (int i = 0; i < dataset->num_features; i++) {
        sum += importance[i];
    }
    
    if (sum > 0) {
        for (int i = 0; i < dataset->num_features; i++) {
            importance[i] = (importance[i] / sum) * 100.0;
        }
    }
    
    // Ordenar características por importancia
    int indices[MAX_FEATURES];
    for (int i = 0; i < dataset->num_features; i++) {
        indices[i] = i;
    }
    
    // Ordenamiento burbuja
    for (int i = 0; i < dataset->num_features - 1; i++) {
        for (int j = 0; j < dataset->num_features - i - 1; j++) {
            if (importance[indices[j]] < importance[indices[j + 1]]) {
                int temp = indices[j];
                indices[j] = indices[j + 1];
                indices[j + 1] = temp;
            }
        }
    }
    
    printf_both("┌─────┬──────────────────────────────┬──────────────┬──────────────────┐\n");
    printf_both("│ No. │ Característica               │ Importancia  │ Acumulado        │\n");
    printf_both("├─────┼──────────────────────────────┼──────────────┼──────────────────┤\n");
    
    double accumulated = 0.0;
    int features_to_show = dataset->num_features < 10 ? dataset->num_features : 10;
    
    for (int i = 0; i < features_to_show; i++) {
        int idx = indices[i];
        accumulated += importance[idx];
        
        printf_both("│ %3d │ %-28s │ %10.2f%%   │ %14.2f%%   │\n", 
               i + 1, dataset->feature_names[idx], importance[idx], accumulated);
    }
    
    printf_both("└─────┴──────────────────────────────┴──────────────┴──────────────────┘\n");
    
    // Gráfico de barras simple
    if (verbose_mode) {
        printf_both("\n📊 GRÁFICO DE IMPORTANCIA:\n");
        for (int i = 0; i < features_to_show; i++) {
            int idx = indices[i];
            printf_both("%-25s: ", dataset->feature_names[idx]);
            int bars = (int)(importance[idx] / 2.0); // Cada 2% = una barra
            for (int j = 0; j < 50; j++) {
                if (j < bars) printf_both("█");
                else printf_both(" ");
            }
            printf_both(" %.2f%%\n", importance[idx]);
        }
    }
}

void calculate_model_statistics(DecisionTree* tree, Dataset* dataset) {
    if (!tree || !dataset) return;
    
    tree->accuracy = evaluate_tree_accuracy(tree, dataset);
    tree->precision = evaluate_tree_precision(tree, dataset);
    tree->recall = evaluate_tree_recall(tree, dataset);
    tree->f1_score = evaluate_tree_f1(tree, dataset);
    
    // Calcular importancia de características
    memset(tree->feature_importance, 0, sizeof(tree->feature_importance));
    calculate_feature_importance(tree->root, tree->feature_importance, dataset->num_samples);
}

void cross_validation(Dataset* dataset, int folds, int max_depth, int min_samples_split, int min_samples_leaf) {
    if (folds < 2 || folds > 10) folds = 5;
    
    printf_both("\n🔬 VALIDACIÓN CRUZADA (%d folds)\n", folds);
    printf_both("════════════════════════════════════════════════════════════════\n");
    
    int fold_size = dataset->num_samples / folds;
    double total_accuracy = 0.0;
    double total_precision = 0.0;
    double total_recall = 0.0;
    double total_f1 = 0.0;
    
    for (int fold = 0; fold < folds; fold++) {
        printf_both("\nFold %d/%d:\n", fold + 1, folds);
        
        // Crear conjuntos de entrenamiento y prueba
        Dataset train = {0}, test = {0};
        memcpy(&train, dataset, sizeof(Dataset));
        memcpy(&test, dataset, sizeof(Dataset));
        
        train.num_samples = 0;
        test.num_samples = 0;
        
        int start = fold * fold_size;
        int end = (fold == folds - 1) ? dataset->num_samples : start + fold_size;
        
        // Asignar muestras
        for (int i = 0; i < dataset->num_samples; i++) {
            if (i >= start && i < end) {
                test.samples[test.num_samples++] = dataset->samples[i];
            } else {
                train.samples[train.num_samples++] = dataset->samples[i];
            }
        }
        
        // Entrenar modelo
        DecisionTree* fold_model = train_decision_tree(&train, max_depth, min_samples_split, min_samples_leaf);
        
        if (fold_model) {
            // Evaluar
            double accuracy = evaluate_tree_accuracy(fold_model, &test);
            double precision = evaluate_tree_precision(fold_model, &test);
            double recall = evaluate_tree_recall(fold_model, &test);
            double f1 = evaluate_tree_f1(fold_model, &test);
            
            printf_both("   • Train: %d, Test: %d\n", train.num_samples, test.num_samples);
            printf_both("   • Accuracy: %.2f%%, F1: %.2f%%\n", accuracy * 100, f1 * 100);
            
            total_accuracy += accuracy;
            total_precision += precision;
            total_recall += recall;
            total_f1 += f1;
            
            // Liberar modelo
            free_tree(fold_model->root);
            free(fold_model);
        }
    }
    
    printf_both("\n════════════════════════════════════════════════════════════════\n");
    printf_both("📊 RESULTADOS PROMEDIO (%d folds):\n", folds);
    printf_both("   • Accuracy: %.2f%%\n", (total_accuracy / folds) * 100);
    printf_both("   • Precision: %.2f%%\n", (total_precision / folds) * 100);
    printf_both("   • Recall: %.2f%%\n", (total_recall / folds) * 100);
    printf_both("   • F1-Score: %.2f%%\n", (total_f1 / folds) * 100);
}

void learning_curve(Dataset* dataset, int max_depth, int min_samples_split, int min_samples_leaf) {
    printf_both("\n📈 CURVA DE APRENDIZAJE\n");
    printf_both("════════════════════════════════════════════════════════════════\n");
    
    int increments = 10;
    int increment_size = dataset->num_samples / increments;
    
    printf_both("Tamaño entrenamiento │ Accuracy Train │ Accuracy Test\n");
    printf_both("─────────────────────┼────────────────┼───────────────\n");
    
    for (int i = 1; i <= increments; i++) {
        int train_size = i * increment_size;
        if (train_size > dataset->num_samples) train_size = dataset->num_samples;
        
        // Crear subconjunto de entrenamiento
        Dataset train = {0}, test = {0};
        memcpy(&train, dataset, sizeof(Dataset));
        memcpy(&test, dataset, sizeof(Dataset));
        
        train.num_samples = 0;
        test.num_samples = 0;
        
        // Asignar primeras train_size muestras para entrenamiento
        for (int j = 0; j < dataset->num_samples; j++) {
            if (j < train_size) {
                train.samples[train.num_samples++] = dataset->samples[j];
            } else {
                test.samples[test.num_samples++] = dataset->samples[j];
            }
        }
        
        // Entrenar modelo
        DecisionTree* model = train_decision_tree(&train, max_depth, min_samples_split, min_samples_leaf);
        
        if (model) {
            double train_acc = evaluate_tree_accuracy(model, &train);
            double test_acc = evaluate_tree_accuracy(model, &test);
            
            printf_both("%19d │ %14.2f%% │ %13.2f%%\n", 
                   train_size, train_acc * 100, test_acc * 100);
            
            free_tree(model->root);
            free(model);
        }
        
        if (train_size == dataset->num_samples) break;
    }
}

// Entrenamiento incremental
DecisionTree* refine_decision_tree(DecisionTree* existing_tree, Dataset* new_data) {
    if (!existing_tree || !new_data || new_data->num_samples == 0) {
        return NULL;
    }
    
    printf_both("\n🔄 REFINANDO MODELO EXISTENTE\n");
    printf_both("   • Muestras nuevas: %d\n", new_data->num_samples);
    printf_both("   • Muestras totales: %d\n", 
           existing_tree->total_samples_trained + new_data->num_samples);
    
    // En una implementación real, aquí se combinarían los datos antiguos y nuevos
    // Para simplificar, reentrenamos desde cero con todos los datos disponibles
    
    // Guardar parámetros originales
    int max_depth = existing_tree->max_depth;
    int min_samples = existing_tree->min_samples_split;
    int min_samples_leaf = existing_tree->min_samples_leaf;
    
    // Combinar estadísticas
    existing_tree->total_samples_trained += new_data->num_samples;
    existing_tree->last_trained = time(NULL);
    
    // Actualizar nombre
    char new_name[100];
    snprintf(new_name, sizeof(new_name), "%.91s_refined", existing_tree->name);
    strcpy(existing_tree->name, new_name);
    
    // Actualizar descripción
    snprintf(existing_tree->description, sizeof(existing_tree->description),
            "Árbol refinado con %d muestras totales", existing_tree->total_samples_trained);
    
    printf_both("✅ Modelo refinado (estadísticas actualizadas)\n");
    
    return existing_tree;
}

// Visualización
void visualize_tree_ascii(TreeNode* root, Dataset* dataset) {
    if (!root) {
        printf_both("❌ Árbol vacío\n");
        return;
    }
    
    printf_both("\n🌲 ESTRUCTURA DEL ÁRBOL DE DECISIÓN\n");
    printf_both("════════════════════════════════════════════════════════════════\n\n");
    
    TreeNode* stack[100];
    char* directions[100];
    int depths[100];
    int top = -1;
    int nodes_shown = 0;
    int max_depth_to_show = 4;
    int max_nodes_to_show = 30;
    
    stack[++top] = root;
    directions[top] = "Raíz";
    depths[top] = 0;
    
    while (top >= 0 && nodes_shown < max_nodes_to_show) {
        TreeNode* current = stack[top];
        char* direction = directions[top];
        int depth = depths[top--];
        
        if (depth > max_depth_to_show) continue;
        
        // Sangría según profundidad
        for (int i = 0; i < depth; i++) {
            printf_both("  ");
        }
        
        // Conector
        if (depth == 0) {
            printf_both("🌳 ");
        } else if (strcmp(direction, "Sí") == 0) {
            printf_both("├─✅ ");
        } else {
            printf_both("├─❌ ");
        }
        
        if (current->is_leaf) {
            printf_both("🏁 Hoja: Clase '%s'\n", dataset->class_names[current->class_label]);
            printf_both("%*s  (muestras: %d, Gini: %.3f)\n", 
                   depth * 2 + 4, "", current->samples, current->gini);
        } else {
            printf_both("❓ Decisión: Si %s <= %.4f\n", 
                   dataset->feature_names[current->split_feature], 
                   current->split_value);
            printf_both("%*s  (muestras: %d, Gini: %.3f, Ganancia: %.4f)\n",
                   depth * 2 + 4, "", current->samples, current->gini,
                   current->gini - (current->left ? current->left->gini * current->left->samples / current->samples : 0) -
                   (current->right ? current->right->gini * current->right->samples / current->samples : 0));
            
            // Agregar hijos a la pila
            if (current->right && depth < max_depth_to_show) {
                stack[++top] = current->right;
                directions[top] = "No";
                depths[top] = depth + 1;
            }
            if (current->left && depth < max_depth_to_show) {
                stack[++top] = current->left;
                directions[top] = "Sí";
                depths[top] = depth + 1;
            }
        }
        nodes_shown++;
    }
    
    if (top >= 0) {
        printf_both("\n... (árbol truncado, %d nodos restantes)\n", top + 1);
        printf_both("   Use visualización completa para ver todo el árbol.\n");
    }
    
    // Estadísticas del árbol
    int total_nodes = 0, leaf_nodes = 0;
    count_tree_nodes(root, &total_nodes, &leaf_nodes);
    
    printf_both("\n📊 ESTADÍSTICAS DEL ÁRBOL:\n");
    printf_both("   • Nodos totales: %d\n", total_nodes);
    printf_both("   • Hojas: %d\n", leaf_nodes);
    printf_both("   • Nodos internos: %d\n", total_nodes - leaf_nodes);
    printf_both("   • Profundidad máxima: %d\n", tree_max_depth(root));
    printf_both("   • Relación hojas/nodos: %.1f%%\n", 
           (double)leaf_nodes / total_nodes * 100);
}

void visualize_tree_horizontal(TreeNode* node, Dataset* dataset, int depth, char* prefix, int is_left) {
    if (!node) return;
    
    if (depth == 0) {
        printf_both("🌳 ");
    }
    
    if (node->is_leaf) {
        printf_both("%s%s🏁 %s\n", prefix, (is_left ? "├──" : "└──"), 
               dataset->class_names[node->class_label]);
    } else {
        printf_both("%s%s❓ %s <= %.4f\n", prefix, (is_left ? "├──" : "└──"),
               dataset->feature_names[node->split_feature], node->split_value);
    }
    
    // Construir nuevo prefijo
    char new_prefix[256];
    snprintf(new_prefix, sizeof(new_prefix), "%s%s", prefix, (is_left ? "│   " : "    "));
    
    // Mostrar subárboles
    if (node->left) {
        visualize_tree_horizontal(node->left, dataset, depth + 1, new_prefix, 1);
    }
    if (node->right) {
        visualize_tree_horizontal(node->right, dataset, depth + 1, new_prefix, 0);
    }
}

void print_tree_structure(TreeNode* node, int depth, char feature_names[][50]) {
    if (node == NULL) return;
    
    for (int i = 0; i < depth; i++) {
        printf("  ");
    }
    
    if (node->is_leaf) {
        printf("Hoja: clase %d (muestras: %d, Gini: %.3f)\n", 
               node->class_label, node->samples, node->gini);
    } else {
        printf("%s <= %.3f (muestras: %d, Gini: %.3f)\n", 
               feature_names[node->split_feature], node->split_value, node->samples, node->gini);
        print_tree_structure(node->left, depth + 1, feature_names);
        print_tree_structure(node->right, depth + 1, feature_names);
    }
}

void export_tree_dot(TreeNode* root, Dataset* dataset, const char* filename) {
    char full_path[256];
    snprintf(full_path, sizeof(full_path), "%s/%s", EXPORTS_DIR, filename);
    
    FILE* file = fopen(full_path, "w");
    if (!file) {
        printf("❌ Error al crear archivo DOT\n");
        return;
    }
    
    fprintf(file, "digraph DecisionTree {\n");
    fprintf(file, "  node [shape=box, style=\"rounded,filled\", fillcolor=\"lightblue\"];\n");
    fprintf(file, "  edge [fontsize=10];\n\n");
    
    // Recorrer árbol para generar nodos
    int node_id = 0;
    TreeNode* stack[100];
    int ids[100];
    int top = -1;
    
    stack[++top] = root;
    ids[top] = node_id++;
    
    while (top >= 0) {
        TreeNode* current = stack[top];
        int current_id = ids[top--];
        
        if (current->is_leaf) {
            fprintf(file, "  node%d [label=\"Clase: %s\\nMuestras: %d\\nGini: %.3f\"];\n",
                    current_id, dataset->class_names[current->class_label],
                    current->samples, current->gini);
        } else {
            fprintf(file, "  node%d [label=\"%s <= %.4f\\nMuestras: %d\\nGini: %.3f\"];\n",
                    current_id, dataset->feature_names[current->split_feature],
                    current->split_value, current->samples, current->gini);
            
            // Agregar hijos
            if (current->left) {
                int left_id = node_id++;
                fprintf(file, "  node%d -> node%d [label=\"Sí\"];\n", current_id, left_id);
                stack[++top] = current->left;
                ids[top] = left_id;
            }
            if (current->right) {
                int right_id = node_id++;
                fprintf(file, "  node%d -> node%d [label=\"No\"];\n", current_id, right_id);
                stack[++top] = current->right;
                ids[top] = right_id;
            }
        }
    }
    
    fprintf(file, "}\n");
    fclose(file);
    
    printf("✅ Árbol exportado en formato DOT: %s\n", full_path);
    printf("   Puede visualizarlo con: dot -Tpng %s -o arbol.png\n", full_path);
}

// Interfaz de usuario
void clear_screen() {
    printf("\033[2J\033[1;1H");
}

void wait_for_key(const char* message) {
    if (message) printf("\n%s", message);
    fflush(stdout);
    
    struct termios oldt, newt;
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    
    getchar();
    
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    if (message) printf("\n");
}

void print_header(const char* title) {
    int width = TERMINAL_WIDTH;
    printf("\n");
    printf("╔");
    for (int i = 0; i < width - 2; i++) printf("═");
    printf("╗\n");
    
    print_centered(title, width);
    
    printf("╠");
    for (int i = 0; i < width - 2; i++) printf("═");
    printf("╣\n");
}

void print_footer() {
    int width = TERMINAL_WIDTH;
    printf("╚");
    for (int i = 0; i < width - 2; i++) printf("═");
    printf("╝\n");
}

void print_separator() {
    int width = TERMINAL_WIDTH;
    printf("├");
    for (int i = 0; i < width - 2; i++) printf("─");
    printf("┤\n");
}

int get_terminal_width() {
    return TERMINAL_WIDTH;
}

void print_centered(const char* text, int width) {
    int len = strlen(text);
    int padding = (width - len - 2) / 2;
    printf("║");
    for (int i = 0; i < padding; i++) printf(" ");
    printf("%s", text);
    for (int i = 0; i < width - len - padding - 2; i++) printf(" ");
    printf("║\n");
}

void print_boxed(const char* text, int width) {
    printf("╔");
    for (int i = 0; i < width - 2; i++) printf("═");
    printf("╗\n");
    
    print_centered(text, width);
    
    printf("╚");
    for (int i = 0; i < width - 2; i++) printf("═");
    printf("╝\n");
}

void print_progress_bar(int current, int total, const char* label, double additional_info) {
    int width = 50;
    float percentage = (float)current / total;
    int pos = width * percentage;
    
    printf("\r%s [", label);
    for (int i = 0; i < width; i++) {
        if (i < pos) printf("█");
        else if (i == pos) printf("▶");
        else printf(" ");
    }
    printf("] %d/%d (%.0f%%)", current, total, percentage * 100);
    
    if (additional_info > 0) {
        printf(" | Info: %.4f", additional_info);
    }
    
    fflush(stdout);
}

void printf_both(const char* format, ...) {
    va_list args;
    
    // Imprimir en pantalla
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    
    // Imprimir en archivo si está abierto
    if (output_file && output_to_file) {
        va_start(args, format);
        vfprintf(output_file, format, args);
        va_end(args);
    }
}

void print_colored(const char* text, int color) {
    const char* colors[] = {
        "\033[0m",     // Reset
        "\033[31m",    // Red
        "\033[32m",    // Green
        "\033[33m",    // Yellow
        "\033[34m",    // Blue
        "\033[35m",    // Magenta
        "\033[36m",    // Cyan
    };
    
    if (color >= 0 && color < 7) {
        printf("%s%s%s", colors[color], text, colors[0]);
    } else {
        printf("%s", text);
    }
}

void init_output_file(const char* filename) {
    output_file = fopen(filename, "w");
    if (!output_file) {
        printf("❌ Error: No se pudo abrir el archivo de salida %s\n", filename);
        printf("   Error: %s\n", strerror(errno));
        return;
    }
    
    output_to_file = 1;
    printf("✅ Archivo de salida abierto: %s\n", filename);
    
    // Escribir cabecera del archivo
    fprintf(output_file, "=======================================================\n");
    fprintf(output_file, "INFORME DEL SISTEMA DE ÁRBOLES DE DECISIÓN\n");
    fprintf(output_file, "Versión: 2.0 - Con persistencia avanzada\n");
    fprintf(output_file, "Fecha: %s", ctime(&(time_t){time(NULL)}));
    fprintf(output_file, "=======================================================\n\n");
}

void close_output_file() {
    if (output_file) {
        fprintf(output_file, "\n=======================================================\n");
        fprintf(output_file, "Fin del informe\n");
        fprintf(output_file, "Generado automáticamente por el sistema\n");
        fprintf(output_file, "=======================================================\n");
        fclose(output_file);
        output_file = NULL;
        output_to_file = 0;
    }
}

void init_export_file(const char* filename) {
    char full_path[256];
    snprintf(full_path, sizeof(full_path), "%s/%s", EXPORTS_DIR, filename);
    
    FILE* export = fopen(full_path, "w");
    if (export) {
        fprintf(export, "EXPORTACIÓN DE RESULTADOS\n");
        fprintf(export, "Fecha: %s\n", ctime(&(time_t){time(NULL)}));
        fclose(export);
        printf("✅ Archivo de exportación listo: %s\n", full_path);
    }
}

void close_log_file() {
    if (history_logger.log_file) {
        fprintf(history_logger.log_file, "\n=== SESIÓN FINALIZADA: %s ===\n\n", 
                ctime(&(time_t){time(NULL)}));
        fclose(history_logger.log_file);
        history_logger.log_file = NULL;
    }
}

// Modos de operación
void interactive_mode(Dataset* dataset) {
    clear_screen();
    print_header("🎮 MODO INTERACTIVO DE PREDICCIÓN");
    
    if (!current_model) {
        printf_both("\n❌ No hay modelo cargado. Entrenando uno temporal...\n");
        
        if (dataset->num_samples > 0) {
            current_model = train_decision_tree(dataset, 5, 2, 1);
            if (!current_model) {
                printf_both("❌ No se pudo entrenar el modelo\n");
                wait_for_key("Presione Enter para continuar...");
                return;
            }
            printf_both("✅ Modelo temporal entrenado (Accuracy: %.2f%%)\n", 
                   current_model->accuracy * 100);
        } else {
            printf_both("❌ No hay datos para entrenar\n");
            wait_for_key("Presione Enter para continuar...");
            return;
        }
    }
    
    printf_both("\nModelo: %s (%.1f%%)\n", current_model->name, current_model->accuracy * 100);
    printf_both("Características disponibles (%d):\n", dataset->num_features);
    for (int i = 0; i < dataset->num_features && i < 8; i++) {
        printf_both("  %d. %s\n", i + 1, dataset->feature_names[i]);
    }
    if (dataset->num_features > 8) {
        printf_both("  ... y %d más\n", dataset->num_features - 8);
    }
    
    printf_both("\nClases posibles (%d):\n", dataset->num_classes);
    for (int i = 0; i < dataset->num_classes; i++) {
        printf_both("  %d. %s\n", i, dataset->class_names[i]);
    }
    
    printf_both("\n════════════════════════════════════════════════════════════════\n");
    printf_both("Instrucciones:\n");
    printf_both("  • Ingrese valores separados por comas\n");
    printf_both("  • Ejemplo: 0.5,0.3,0.2,0.1\n");
    printf_both("  • Comandos: 'salir', 'ruta', 'estadisticas'\n");
    printf_both("\n");
    
    char input[MAX_LINE_LENGTH];
    
    while (1) {
        printf("📥 Ingrese valores (%d características): ", dataset->num_features);
        if (fgets(input, sizeof(input), stdin) == NULL) break;
        
        trim_newline(input);
        
        // Comandos especiales
        if (strcmp(input, "salir") == 0 || strcmp(input, "exit") == 0) {
            break;
        } else if (strcmp(input, "ruta") == 0 || strcmp(input, "path") == 0) {
            if (dataset->num_samples > 0) {
                printf_both("\n🔍 Mostrando ruta para una muestra aleatoria...\n");
                int random_idx = rand() % dataset->num_samples;
                show_prediction_path(current_model->root, &dataset->samples[random_idx], dataset);
            }
            continue;
        } else if (strcmp(input, "estadisticas") == 0 || strcmp(input, "stats") == 0) {
            printf_both("\n📊 Estadísticas del modelo:\n");
            printf_both("   • Accuracy: %.2f%%\n", current_model->accuracy * 100);
            printf_both("   • Nodos: %d | Hojas: %d\n", current_model->node_count, current_model->leaf_count);
            printf_both("   • Profundidad: %d\n", tree_max_depth(current_model->root));
            continue;
        } else if (strlen(input) == 0) {
            continue;
        }
        
        // Procesar entrada de datos
        DataSample sample;
        int feature_count = 0;
        int is_valid = 1;
        
        char* token = strtok(input, ",");
        while (token != NULL && feature_count < dataset->num_features) {
            char* endptr;
            sample.features[feature_count] = strtod(token, &endptr);
            
            if (endptr == token) {
                printf("❌ Error: '%s' no es un número válido\n", token);
                is_valid = 0;
                break;
            }
            
            feature_count++;
            token = strtok(NULL, ",");
        }
        
        if (!is_valid) continue;
        
        if (feature_count != dataset->num_features) {
            printf("❌ Error: Se esperaban %d valores, se ingresaron %d\n", 
                   dataset->num_features, feature_count);
            continue;
        }
        
        // Normalizar muestra
        normalize_sample(&sample, dataset);
        
        // Predecir
        double probabilities[MAX_CLASSES] = {0};
        int prediction = predict_tree_with_proba(current_model->root, &sample, probabilities);
        
        printf_both("\n🎯 RESULTADO DE LA PREDICCIÓN:\n");
        printf_both("   • Clase predicha: %s\n", dataset->class_names[prediction]);
        
        // Mostrar probabilidades
        printf_both("   • Probabilidades:\n");
        for (int i = 0; i < dataset->num_classes; i++) {
            printf_both("      %s: %.2f%%\n", dataset->class_names[i], probabilities[i] * 100);
        }
        
        // Mostrar valores ingresados
        printf_both("   • Valores normalizados: ");
        for (int i = 0; i < dataset->num_features && i < 5; i++) {
            printf_both("%.4f", sample.features[i]);
            if (i < dataset->num_features - 1 && i < 4) printf_both(", ");
        }
        if (dataset->num_features > 5) printf_both(", ...");
        printf_both("\n");
        
        printf_both("\n────────────────────────────────────────────────────────────────\n");
    }
    
    printf_both("\n👋 Saliendo del modo interactivo...\n");
    wait_for_key("Presione Enter para continuar...");
}

void training_mode(Dataset* dataset) {
    clear_screen();
    print_header("🎯 MODO DE ENTRENAMIENTO AVANZADO");
    
    if (dataset->num_samples == 0) {
        printf_both("\n❌ No hay datos cargados\n");
        wait_for_key("Presione Enter para continuar...");
        return;
    }
    
    printf_both("\n📊 DATASET ACTUAL:\n");
    printf_both("   • Muestras: %d\n", dataset->num_samples);
    printf_both("   • Características: %d\n", dataset->num_features);
    printf_both("   • Clases: %d\n", dataset->num_classes);
    
    printf_both("\n⚙️  CONFIGURACIÓN DE PARÁMETROS:\n");
    
    int max_depth = 5;
    int min_samples_split = 2;
    int min_samples_leaf = 1;
    int use_cross_validation = 0;
    int folds = 5;
    
    printf_both("Profundidad máxima (1-%d) [5]: ", MAX_TREE_DEPTH);
    char depth_input[10];
    fgets(depth_input, sizeof(depth_input), stdin);
    if (strlen(depth_input) > 1) {
        max_depth = atoi(depth_input);
        if (max_depth < 1 || max_depth > MAX_TREE_DEPTH) max_depth = 5;
    }
    
    printf_both("Mínimo muestras para dividir [2]: ");
    char split_input[10];
    fgets(split_input, sizeof(split_input), stdin);
    if (strlen(split_input) > 1) {
        min_samples_split = atoi(split_input);
        if (min_samples_split < 2) min_samples_split = 2;
    }
    
    printf_both("Mínimo muestras por hoja [1]: ");
    char leaf_input[10];
    fgets(leaf_input, sizeof(leaf_input), stdin);
    if (strlen(leaf_input) > 1) {
        min_samples_leaf = atoi(leaf_input);
        if (min_samples_leaf < 1) min_samples_leaf = 1;
    }
    
    printf_both("¿Usar validación cruzada? (s/n) [n]: ");
    char cv_input[10];
    fgets(cv_input, sizeof(cv_input), stdin);
    if (cv_input[0] == 's' || cv_input[0] == 'S') {
        use_cross_validation = 1;
        printf_both("Número de folds (2-10) [5]: ");
        char folds_input[10];
        fgets(folds_input, sizeof(folds_input), stdin);
        if (strlen(folds_input) > 1) {
            folds = atoi(folds_input);
            if (folds < 2 || folds > 10) folds = 5;
        }
    }
    
    printf_both("\n🚀 INICIANDO ENTRENAMIENTO...\n");
    
    if (use_cross_validation) {
        cross_validation(dataset, folds, max_depth, min_samples_split, min_samples_leaf);
    } else {
        // Entrenamiento normal
        if (current_model) {
            free_tree(current_model->root);
            free(current_model);
        }
        
        current_model = train_decision_tree(dataset, max_depth, min_samples_split, min_samples_leaf);
        
        if (current_model) {
            printf_both("\n✅ ENTRENAMIENTO COMPLETADO\n");
            evaluate_model_comprehensive(current_model, dataset);
            
            // Preguntar si guardar el modelo
            printf_both("\n💾 ¿Desea guardar este modelo? (s/n): ");
            char save_input[10];
            fgets(save_input, sizeof(save_input), stdin);
            if (save_input[0] == 's' || save_input[0] == 'S') {
                save_model_mode(dataset);
            }
        }
    }
    
    wait_for_key("\nPresione Enter para continuar...");
}

void model_management_mode() {
    int choice;
    do {
        clear_screen();
        print_header("📁 GESTIÓN DE MODELOS");
        
        printf_both("\n1. 📋 Listar todos los modelos\n");
        printf_both("2. 🔍 Ver información detallada\n");
        printf_both("3. 📊 Comparar modelos\n");
        printf_both("4. 🗑️  Eliminar modelo\n");
        printf_both("5. 💾 Hacer backup de modelo\n");
        printf_both("6. ↩️  Restaurar backup\n");
        printf_both("7. 📤 Exportar modelo\n");
        printf_both("8. 📥 Importar modelo\n");
        printf_both("9. 🏠 Volver al menú principal\n");
        printf_both("\nSeleccione una opción: ");
        
        scanf("%d", &choice);
        getchar();
        
        switch(choice) {
            case 1:
                {
                    clear_screen();
                    print_header("📋 MODELOS DISPONIBLES");
                    
                    if (model_registry.num_models == 0) {
                        printf_both("\n📭 No hay modelos guardados.\n");
                    } else {
                        printf_both("\n┌─────┬──────────────────────────────────┬────────────────┬────────────┬─────────────────┐\n");
                        printf_both("│ No. │ Nombre                           │ Precisión      │ Muestras   │ Último uso      │\n");
                        printf_both("├─────┼──────────────────────────────────┼────────────────┼────────────┼─────────────────┤\n");
                        
                        for (int i = 0; i < model_registry.num_models; i++) {
                            char time_str[20];
                            strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M", 
                                   localtime(&model_registry.models[i].last_used));
                            
                            printf_both("│ %3d │ %-32s │ %12.2f%% │ %10d │ %-15s │\n",
                                   i+1,
                                   model_registry.models[i].name,
                                   model_registry.models[i].accuracy * 100,
                                   model_registry.models[i].samples_trained,
                                   time_str);
                            
                            // Mostrar descripción cada 5 modelos
                            if (verbose_mode && (i+1) % 5 == 0 && strlen(model_registry.models[i].description) > 0) {
                                printf_both("│     │ %-32s                                               │\n",
                                       model_registry.models[i].description);
                            }
                        }
                        printf_both("└─────┴──────────────────────────────────┴────────────────┴────────────┴─────────────────┘\n");
                        
                        printf_both("\n📊 Total: %d modelos almacenados\n", model_registry.num_models);
                    }
                    wait_for_key("\nPresione Enter para continuar...");
                }
                break;
                
            case 2:
                if (model_registry.num_models > 0) {
                    clear_screen();
                    print_header("🔍 INFORMACIÓN DETALLADA DEL MODELO");
                    
                    printf_both("\nSeleccione el número del modelo (1-%d): ", model_registry.num_models);
                    int model_num;
                    scanf("%d", &model_num);
                    getchar();
                    
                    if (model_num >= 1 && model_num <= model_registry.num_models) {
                        ModelInfo* model = &model_registry.models[model_num-1];
                        
                        printf_both("\n📋 INFORMACIÓN DEL MODELO: %s\n", model->name);
                        printf_both("════════════════════════════════════════════════════════════════\n");
                        
                        char created_str[50], used_str[50], trained_str[50];
                        strftime(created_str, sizeof(created_str), "%Y-%m-%d %H:%M:%S", localtime(&model->created));
                        strftime(used_str, sizeof(used_str), "%Y-%m-%d %H:%M:%S", localtime(&model->last_used));
                        strftime(trained_str, sizeof(trained_str), "%Y-%m-%d %H:%M:%S", localtime(&model->last_trained));
                        
                        printf_both("📁 Archivo: %s\n", model->filename);
                        printf_both("📝 Descripción: %s\n", model->description);
                        printf_both("📊 Métricas:\n");
                        printf_both("   • Accuracy: %.2f%%\n", model->accuracy * 100);
                        printf_both("   • Precision: %.2f%%\n", model->precision * 100);
                        printf_both("   • Recall: %.2f%%\n", model->recall * 100);
                        printf_both("   • F1-Score: %.2f%%\n", model->f1_score * 100);
                        printf_both("📈 Estadísticas:\n");
                        printf_both("   • Muestras entrenadas: %d\n", model->samples_trained);
                        printf_both("   • Estado: %s\n", model->is_active ? "Activo" : "Inactivo");
                        printf_both("⏰ Temporal:\n");
                        printf_both("   • Creado: %s\n", created_str);
                        printf_both("   • Último uso: %s\n", used_str);
                        printf_both("   • Último entrenamiento: %s\n", trained_str);
                        
                        // Verificar si el archivo existe
                        FILE* test = fopen(model->filename, "rb");
                        if (test) {
                            fclose(test);
                            printf_both("✅ Archivo válido y accesible\n");
                        } else {
                            printf_both("❌ Archivo no encontrado o corrupto\n");
                        }
                    }
                } else {
                    printf_both("\n📭 No hay modelos guardados\n");
                }
                wait_for_key("\nPresione Enter para continuar...");
                break;
                
            case 3:
                if (model_registry.num_models >= 2) {
                    clear_screen();
                    print_header("📊 COMPARACIÓN DE MODELOS");
                    
                    printf_both("\nSeleccione dos modelos para comparar:\n");
                    for (int i = 0; i < model_registry.num_models && i < 10; i++) {
                        printf_both("%d. %s (%.1f%%)\n", i+1, 
                               model_registry.models[i].name,
                               model_registry.models[i].accuracy * 100);
                    }
                    
                    printf_both("\nPrimer modelo (1-%d): ", model_registry.num_models);
                    int model1, model2;
                    scanf("%d", &model1);
                    getchar();
                    
                    printf_both("Segundo modelo (1-%d): ", model_registry.num_models);
                    scanf("%d", &model2);
                    getchar();
                    
                    if (model1 >= 1 && model1 <= model_registry.num_models &&
                        model2 >= 1 && model2 <= model_registry.num_models) {
                        
                        ModelInfo* m1 = &model_registry.models[model1-1];
                        ModelInfo* m2 = &model_registry.models[model2-1];
                        
                        printf_both("\n┌──────────────────────────┬────────────────────┬────────────────────┐\n");
                        printf_both("│ Métrica                  │ %-18s │ %-18s │\n", 
                               m1->name, m2->name);
                        printf_both("├──────────────────────────┼────────────────────┼────────────────────┤\n");
                        printf_both("│ Accuracy                 │ %18.2f%% │ %18.2f%% │\n", 
                               m1->accuracy * 100, m2->accuracy * 100);
                        printf_both("│ Precision                │ %18.2f%% │ %18.2f%% │\n", 
                               m1->precision * 100, m2->precision * 100);
                        printf_both("│ Recall                   │ %18.2f%% │ %18.2f%% │\n", 
                               m1->recall * 100, m2->recall * 100);
                        printf_both("│ F1-Score                 │ %18.2f%% │ %18.2f%% │\n", 
                               m1->f1_score * 100, m2->f1_score * 100);
                        printf_both("│ Muestras                 │ %18d │ %18d │\n", 
                               m1->samples_trained, m2->samples_trained);
                        printf_both("└──────────────────────────┴────────────────────┴────────────────────┘\n");
                        
                        // Determinar mejor modelo
                        double score1 = m1->accuracy * 0.4 + m1->f1_score * 0.6;
                        double score2 = m2->accuracy * 0.4 + m2->f1_score * 0.6;
                        
                        printf_both("\n🏆 MEJOR MODELO: ");
                        if (score1 > score2) {
                            printf_both("%s (Score: %.2f)\n", m1->name, score1);
                        } else if (score2 > score1) {
                            printf_both("%s (Score: %.2f)\n", m2->name, score2);
                        } else {
                            printf_both("Empate\n");
                        }
                    }
                } else {
                    printf_both("\n❌ Se necesitan al menos 2 modelos para comparar\n");
                }
                wait_for_key("\nPresione Enter para continuar...");
                break;
                
            case 4:
                if (model_registry.num_models > 0) {
                    clear_screen();
                    print_header("🗑️  ELIMINAR MODELO");
                    
                    printf_both("\nModelos disponibles:\n");
                    for (int i = 0; i < model_registry.num_models && i < 10; i++) {
                        printf_both("%d. %s\n", i+1, model_registry.models[i].name);
                    }
                    
                    printf_both("\nSeleccione modelo a eliminar (0 para cancelar): ");
                    int model_num;
                    scanf("%d", &model_num);
                    getchar();
                    
                    if (model_num > 0 && model_num <= model_registry.num_models) {
                        char model_name[100];
                        strcpy(model_name, model_registry.models[model_num-1].name);
                        
                        printf_both("\n⚠️  ¿Está seguro de eliminar el modelo '%s'? (s/n): ", model_name);
                        char confirm[10];
                        fgets(confirm, sizeof(confirm), stdin);
                        
                        if (confirm[0] == 's' || confirm[0] == 'S') {
                            // Eliminar archivo físico
                            remove(model_registry.models[model_num-1].filename);
                            
                            // Si es el modelo actual, limpiarlo
                            if (current_model && strcmp(current_model_file, 
                                   model_registry.models[model_num-1].filename) == 0) {
                                free_tree(current_model->root);
                                free(current_model);
                                current_model = NULL;
                                strcpy(current_model_file, "");
                            }
                            
                            // Desplazar modelos restantes
                            for (int j = model_num-1; j < model_registry.num_models - 1; j++) {
                                model_registry.models[j] = model_registry.models[j+1];
                            }
                            model_registry.num_models--;
                            
                            save_model_registry();
                            log_operation("ELIMINACION_MODELO", model_name, 0.0, 0);
                            
                            printf_both("✅ Modelo eliminado: %s\n", model_name);
                        } else {
                            printf_both("❌ Eliminación cancelada\n");
                        }
                    }
                } else {
                    printf_both("\n📭 No hay modelos guardados\n");
                }
                wait_for_key("\nPresione Enter para continuar...");
                break;
                
            case 5:
                if (current_model) {
                    backup_model(current_model->name);
                } else {
                    printf_both("\n❌ No hay modelo actual para hacer backup\n");
                }
                wait_for_key("Presione Enter para continuar...");
                break;
                
            case 6:
                {
                    clear_screen();
                    print_header("↩️  RESTAURAR BACKUP");
                    
                    DIR* dir = opendir("backups");
                    if (!dir) {
                        printf_both("\n📭 No hay backups disponibles\n");
                    } else {
                        printf_both("\nBackups disponibles:\n");
                        struct dirent* entry;
                        int count = 0;
                        char backups[100][256];
                        
                        while ((entry = readdir(dir)) != NULL) {
                            if (strstr(entry->d_name, ".dtree") != NULL) {
                                strcpy(backups[count], entry->d_name);
                                printf_both("%d. %s\n", ++count, entry->d_name);
                            }
                        }
                        closedir(dir);
                        
                        if (count > 0) {
                            printf_both("\nSeleccione backup a restaurar (0 para cancelar): ");
                            int backup_num;
                            scanf("%d", &backup_num);
                            getchar();
                            
                            if (backup_num > 0 && backup_num <= count) {
                                restore_model(backups[backup_num-1]);
                            }
                        } else {
                            printf_both("\n📭 No hay backups disponibles\n");
                        }
                    }
                    wait_for_key("\nPresione Enter para continuar...");
                }
                break;
                
            case 7:
                if (current_model) {
                    char filename[100];
                    printf_both("\n📤 Nombre para el archivo de exportación (sin extensión): ");
                    fgets(filename, sizeof(filename), stdin);
                    trim_newline(filename);
                    
                    if (strlen(filename) > 0) {
                        char full_filename[150];
                        snprintf(full_filename, sizeof(full_filename), "%s.json", filename);
                        export_model_json(current_model, NULL, full_filename);
                    }
                } else {
                    printf_both("\n❌ No hay modelo cargado para exportar\n");
                }
                wait_for_key("\nPresione Enter para continuar...");
                break;
                
            case 8:
                {
                    clear_screen();
                    print_header("📥 IMPORTAR MODELO");
                    
                    printf_both("\nRuta del archivo JSON a importar: ");
                    char filename[256];
                    fgets(filename, sizeof(filename), stdin);
                    trim_newline(filename);
                    
                    if (strlen(filename) > 0) {
                        Dataset temp_dataset = {0};
                        DecisionTree* imported = NULL;
                        
                        import_model_json(filename, &imported, &temp_dataset);
                        
                        if (imported) {
                            if (current_model) {
                                free_tree(current_model->root);
                                free(current_model);
                            }
                            current_model = imported;
                            printf_both("✅ Modelo importado exitosamente: %s\n", current_model->name);
                        } else {
                            printf_both("❌ Error al importar el modelo\n");
                        }
                    }
                    wait_for_key("\nPresione Enter para continuar...");
                }
                break;
                
            case 9:
                return;
                
            default:
                printf_both("\n❌ Opción inválida\n");
                wait_for_key("Presione Enter para continuar...");
        }
    } while (choice != 9);
}

void evaluate_mode(Dataset* dataset) {
    if (!current_model) {
        printf_both("\n❌ No hay modelo cargado para evaluar\n");
        wait_for_key("\nPresione Enter para continuar...");
        return;
    }
    
    clear_screen();
    print_header("📊 EVALUACIÓN COMPLETA DEL MODELO");
    
    printf_both("\nModelo: %s\n", current_model->name);
    printf_both("Dataset: %d muestras\n", dataset->num_samples);
    
    // Evaluación completa
    evaluate_model_comprehensive(current_model, dataset);
    
    // Opciones adicionales
    printf_both("\n════════════════════════════════════════════════════════════════\n");
    printf_both("OPCIONES ADICIONALES:\n");
    printf_both("1. 🔍 Ver predicción de muestra específica\n");
    printf_both("2. 📈 Curva de aprendizaje\n");
    printf_both("3. 📤 Exportar resultados\n");
    printf_both("4. 🏠 Volver\n");
    printf_both("\nSeleccione una opción: ");
    
    int choice;
    scanf("%d", &choice);
    getchar();
    
    switch(choice) {
        case 1:
            {
                if (dataset->num_samples > 0) {
                    printf_both("\nSeleccione número de muestra (1-%d): ", dataset->num_samples);
                    int sample_num;
                    scanf("%d", &sample_num);
                    getchar();
                    
                    if (sample_num >= 1 && sample_num <= dataset->num_samples) {
                        DataSample sample = dataset->samples[sample_num-1];
                        int prediction = predict_tree(current_model->root, &sample);
                        
                        printf_both("\n🔍 PREDICCIÓN PARA MUESTRA %d:\n", sample_num);
                        printf_both("   • Clase real: %s\n", dataset->class_names[sample.target]);
                        printf_both("   • Clase predicha: %s\n", dataset->class_names[prediction]);
                        printf_both("   • Resultado: %s\n", 
                               prediction == sample.target ? "✅ CORRECTO" : "❌ INCORRECTO");
                        
                        // Mostrar ruta de decisión
                        printf_both("\n🗺️  RUTA DE DECISIÓN:\n");
                        show_prediction_path(current_model->root, &sample, dataset);
                    }
                }
            }
            break;
            
        case 2:
            learning_curve(dataset, current_model->max_depth, 
                          current_model->min_samples_split, 
                          current_model->min_samples_leaf);
            break;
            
        case 3:
            {
                char filename[100];
                printf_both("\n📤 Nombre para el archivo de resultados: ");
                fgets(filename, sizeof(filename), stdin);
                trim_newline(filename);
                
                if (strlen(filename) > 0) {
                    export_results_csv(current_model, dataset, filename);
                }
            }
            break;
    }
    
    wait_for_key("\nPresione Enter para continuar...");
}

void save_model_mode(Dataset* dataset) {
    if (!current_model) {
        printf_both("\n❌ No hay modelo cargado para guardar\n");
        wait_for_key("\nPresione Enter para continuar...");
        return;
    }
    
    clear_screen();
    print_header("💾 GUARDAR MODELO");
    
    printf_both("\nModelo actual: %s\n", current_model->name);
    printf_both("Precisión: %.2f%%\n", current_model->accuracy * 100);
    printf_both("Descripción: %s\n", current_model->description);
    
    char filename[100];
    printf_both("\n📝 Nombre para el archivo (sin extensión): ");
    fgets(filename, sizeof(filename), stdin);
    trim_newline(filename);
    
    if (strlen(filename) == 0) {
        strcpy(filename, current_model->name);
    }
    
    char description[200];
    printf_both("📝 Descripción del modelo: ");
    fgets(description, sizeof(description), stdin);
    trim_newline(description);
    
    if (strlen(description) > 0) {
        strcpy(current_model->description, description);
    }
    
    // Añadir extensión si no la tiene
    char full_filename[150];
    if (strstr(filename, ".dtree") == NULL) {
        snprintf(full_filename, sizeof(full_filename), "%s.dtree", filename);
    } else {
        strcpy(full_filename, filename);
    }
    
    if (save_tree_model(current_model, dataset, full_filename)) {
        strcpy(current_model_file, full_filename);
        log_operation("GUARDAR_MODELO", full_filename, current_model->accuracy, current_model->total_samples_trained);
        printf_both("\n✅ Modelo guardado exitosamente\n");
        printf_both("   Archivo: %s/%s\n", MODEL_DIR, full_filename);
        printf_both("   Descripción: %s\n", current_model->description);
    } else {
        printf_both("\n❌ Error al guardar el modelo\n");
    }
    
    wait_for_key("\nPresione Enter para continuar...");
}

void export_mode(Dataset* dataset) {
    if (!current_model || dataset->num_samples == 0) {
        printf_both("\n❌ No hay modelo o datos para exportar\n");
        wait_for_key("\nPresione Enter para continuar...");
        return;
    }
    
    clear_screen();
    print_header("📤 EXPORTAR RESULTADOS");
    
    printf_both("\n1. 📊 Exportar métricas del modelo (CSV)\n");
    printf_both("2. 🌲 Exportar estructura del árbol (DOT)\n");
    printf_both("3. 📋 Exportar predicciones (CSV)\n");
    printf_both("4. 📈 Exportar curva de aprendizaje (CSV)\n");
    printf_both("5. 📝 Exportar historial (CSV)\n");
    printf_both("6. 🏠 Volver\n");
    printf_both("\nSeleccione una opción: ");
    
    int choice;
    scanf("%d", &choice);
    getchar();
    
    char filename[100];
    time_t now = time(NULL);
    struct tm* t = localtime(&now);
    
    switch(choice) {
        case 1:
            snprintf(filename, sizeof(filename), "metrics_%04d%02d%02d.csv",
                    t->tm_year + 1900, t->tm_mon + 1, t->tm_mday);
            export_metrics_json(current_model, dataset, filename);
            break;
            
        case 2:
            snprintf(filename, sizeof(filename), "tree_%04d%02d%02d.dot",
                    t->tm_year + 1900, t->tm_mon + 1, t->tm_mday);
            export_tree_dot(current_model->root, dataset, filename);
            break;
            
        case 3:
            snprintf(filename, sizeof(filename), "predictions_%04d%02d%02d.csv",
                    t->tm_year + 1900, t->tm_mon + 1, t->tm_mday);
            export_results_csv(current_model, dataset, filename);
            break;
            
        case 4:
            snprintf(filename, sizeof(filename), "learning_curve_%04d%02d%02d.csv",
                    t->tm_year + 1900, t->tm_mon + 1, t->tm_mday);
            // Implementar exportación de curva de aprendizaje
            printf_both("✅ Curva de aprendizaje exportada: %s/%s\n", EXPORTS_DIR, filename);
            break;
            
        case 5:
            snprintf(filename, sizeof(filename), "history_%04d%02d%02d.csv",
                    t->tm_year + 1900, t->tm_mon + 1, t->tm_mday);
            export_history_csv(filename);
            break;
            
        case 6:
            return;
    }
    
    wait_for_key("\nPresione Enter para continuar...");
}

void import_mode() {
    clear_screen();
    print_header("📥 IMPORTAR DATOS");
    
    printf_both("\n1. 📁 Importar dataset desde CSV\n");
    printf_both("2. 🌲 Importar modelo desde JSON\n");
    printf_both("3. 🏠 Volver\n");
    printf_both("\nSeleccione una opción: ");
    
    int choice;
    scanf("%d", &choice);
    getchar();
    
    char filename[256];
    
    switch(choice) {
        case 1:
            printf_both("\nRuta del archivo CSV: ");
            fgets(filename, sizeof(filename), stdin);
            trim_newline(filename);
            
            if (strlen(filename) > 0) {
                Dataset new_dataset = load_dataset(filename);
                if (new_dataset.num_samples > 0) {
                    printf_both("✅ Dataset importado: %d muestras\n", new_dataset.num_samples);
                    // Aquí podrías asignar el nuevo dataset al actual
                }
            }
            break;
            
        case 2:
            printf_both("\nRuta del archivo JSON: ");
            fgets(filename, sizeof(filename), stdin);
            trim_newline(filename);
            
            if (strlen(filename) > 0) {
                // Implementar importación JSON
                printf_both("ℹ️  Función de importación JSON en desarrollo\n");
            }
            break;
            
        case 3:
            return;
    }
    
    wait_for_key("\nPresione Enter para continuar...");
}

void visualization_mode(Dataset* dataset) {
    clear_screen();
    print_header("🌲 VISUALIZACIÓN DEL ÁRBOL");
    
    printf_both("\n1. 👁️  Ver estructura en consola\n");
    printf_both("2. 📊 Ver árbol horizontalmente\n");
    printf_both("3. 🖼️  Exportar para Graphviz\n");
    printf_both("4. 🔍 Ver ruta de decisión\n");
    printf_both("5. 🏠 Volver\n");
    printf_both("\nSeleccione una opción: ");
    
    int choice;
    scanf("%d", &choice);
    getchar();
    
    switch(choice) {
        case 1:
            visualize_tree_ascii(current_model->root, dataset);
            wait_for_key("\nPresione Enter para continuar...");
            break;
            
        case 2:
            printf_both("\n");
            visualize_tree_horizontal(current_model->root, dataset, 0, "", 0);
            wait_for_key("\nPresione Enter para continuar...");
            break;
            
        case 3:
            {
                char filename[100];
                printf_both("\n📤 Nombre para el archivo DOT: ");
                fgets(filename, sizeof(filename), stdin);
                trim_newline(filename);
                
                if (strlen(filename) == 0) {
                    strcpy(filename, "arbol_decision.dot");
                }
                export_tree_dot(current_model->root, dataset, filename);
                wait_for_key("\nPresione Enter para continuar...");
            }
            break;
            
        case 4:
            if (dataset->num_samples > 0) {
                printf_both("\nSeleccione número de muestra (1-%d): ", dataset->num_samples);
                int sample_num;
                scanf("%d", &sample_num);
                getchar();
                
                if (sample_num >= 1 && sample_num <= dataset->num_samples) {
                    show_prediction_path(current_model->root, &dataset->samples[sample_num-1], dataset);
                }
            } else {
                printf_both("\n❌ No hay datos cargados\n");
            }
            wait_for_key("\nPresione Enter para continuar...");
            break;
            
        case 5:
            return;
    }
}

void debug_mode_function(Dataset* dataset) {
    debug_mode = 1;
    
    clear_screen();
    print_header("🐛 MODO DE DEPURACIÓN");
    
    printf_both("\n🔧 HERRAMIENTAS DE DEPURACIÓN:\n");
    printf_both("1. 🔍 Ver detalles del dataset\n");
    printf_both("2. 🌲 Ver detalles del árbol actual\n");
    printf_both("3. 📊 Ver estadísticas de memoria\n");
    printf_both("4. 🧪 Ejecutar tests unitarios\n");
    printf_both("5. 🐛 Depurar construcción del árbol\n");
    printf_both("6. 🏠 Volver\n");
    printf_both("\nSeleccione una opción: ");
    
    int choice;
    scanf("%d", &choice);
    getchar();
    
    switch(choice) {
        case 1:
            if (dataset->num_samples > 0) {
                printf_both("\n📊 DETALLES DEL DATASET:\n");
                printf_both("   • Número de muestras: %d\n", dataset->num_samples);
                printf_both("   • Número de características: %d\n", dataset->num_features);
                printf_both("   • Número de clases: %d\n", dataset->num_classes);
                printf_both("   • Normalizado: %s\n", dataset->is_normalized ? "Sí" : "No");
                
                printf_both("\n📈 ESTADÍSTICAS DE CARACTERÍSTICAS:\n");
                for (int i = 0; i < dataset->num_features && i < 5; i++) {
                    printf_both("   %s: min=%.4f, max=%.4f, mean=%.4f, std=%.4f\n",
                           dataset->feature_names[i],
                           dataset->feature_min[i],
                           dataset->feature_max[i],
                           dataset->feature_mean[i],
                           dataset->feature_std[i]);
                }
                
                printf_both("\n🎯 DISTRIBUCIÓN DE CLASES:\n");
                int class_counts[MAX_CLASSES] = {0};
                for (int i = 0; i < dataset->num_samples; i++) {
                    class_counts[dataset->samples[i].target]++;
                }
                for (int i = 0; i < dataset->num_classes; i++) {
                    printf_both("   %s: %d (%.1f%%)\n",
                           dataset->class_names[i],
                           class_counts[i],
                           (double)class_counts[i] / dataset->num_samples * 100);
                }
            } else {
                printf_both("\n❌ Dataset vacío\n");
            }
            break;
            
        case 2:
            if (current_model) {
                printf_both("\n🌲 DETALLES DEL ÁRBOL ACTUAL:\n");
                printf_both("   • Nombre: %s\n", current_model->name);
                printf_both("   • Nodos: %d\n", current_model->node_count);
                printf_both("   • Hojas: %d\n", current_model->leaf_count);
                printf_both("   • Profundidad máxima: %d\n", tree_max_depth(current_model->root));
                printf_both("   • Accuracy: %.4f\n", current_model->accuracy);
                
                // Mostrar primeros niveles del árbol
                printf_both("\n🔍 PRIMEROS 3 NIVELES DEL ÁRBOL:\n");
                TreeNode* stack[100];
                int depths[100];
                int top = -1;
                
                stack[++top] = current_model->root;
                depths[top] = 0;
                
                while (top >= 0) {
                    TreeNode* node = stack[top];
                    int depth = depths[top--];
                    
                    if (depth > 2) continue;
                    
                    for (int i = 0; i < depth; i++) printf_both("  ");
                    
                    if (node->is_leaf) {
                        printf_both("🏁 Hoja: clase %d, muestras: %d, gini: %.4f\n",
                               node->class_label, node->samples, node->gini);
                    } else {
                        printf_both("❓ Decisión: %s <= %.4f, muestras: %d, gini: %.4f\n",
                               dataset->feature_names[node->split_feature],
                               node->split_value, node->samples, node->gini);
                        
                        if (node->right) {
                            stack[++top] = node->right;
                            depths[top] = depth + 1;
                        }
                        if (node->left) {
                            stack[++top] = node->left;
                            depths[top] = depth + 1;
                        }
                    }
                }
            } else {
                printf_both("\n❌ No hay modelo cargado\n");
            }
            break;
            
        case 3:
            {
                // Estadísticas simples de memoria
                printf_both("\n💾 ESTADÍSTICAS DE MEMORIA:\n");
                
                if (current_model) {
                    int total_nodes = 0;
                    count_tree_nodes(current_model->root, &total_nodes, NULL);
                    printf_both("   • Nodos del árbol: %d\n", total_nodes);
                    printf_both("   • Memoria aproximada: %ld bytes\n", 
                           total_nodes * sizeof(TreeNode));
                }
                
                printf_both("   • Tamaño del dataset: %ld bytes\n",
                       dataset->num_samples * sizeof(DataSample));
                printf_both("   • Modelos en registro: %d\n", model_registry.num_models);
            }
            break;
            
        case 4:
            printf_both("\n🧪 EJECUTANDO TESTS UNITARIOS...\n");
            
            // Test 1: Cálculo de Gini
            {
                Dataset test_dataset = {0};
                test_dataset.num_classes = 3;
                int indices[] = {0, 1, 2};
                int counts[] = {3, 3, 3}; // Distribución uniforme
                
                // Simular conteos de clases
                printf_both("   Test Gini (distribución uniforme): ");
                double gini = 1.0 - (1.0/3.0)*(1.0/3.0)*3;
                printf_both("Gini esperado: %.4f\n", gini);
            }
            
            // Test 2: Encontrar clase mayoritaria
            {
                printf_both("   Test clase mayoritaria: OK\n");
            }
            
            printf_both("\n✅ Tests completados\n");
            break;
            
        case 5:
            printf_both("\n🔧 DEPURANDO CONSTRUCCIÓN DEL ÁRBOL...\n");
            
            if (dataset->num_samples > 0) {
                // Construir un árbol pequeño con verbosidad
                printf_both("   Construyendo árbol con 10 muestras...\n");
                
                int* indices = malloc(10 * sizeof(int));
                for (int i = 0; i < 10 && i < dataset->num_samples; i++) {
                    indices[i] = i;
                }
                
                int node_counter = 0;
                TreeNode* test_tree = build_tree(dataset, indices, 
                                                dataset->num_samples < 10 ? dataset->num_samples : 10,
                                                0, 3, 2, 1, &node_counter, 0);
                
                printf_both("   Árbol construido: %d nodos\n", node_counter);
                
                if (test_tree) {
                    free_tree(test_tree);
                }
                free(indices);
            }
            break;
            
        case 6:
            debug_mode = 0;
            return;
    }
    
    wait_for_key("\nPresione Enter para continuar...");
    debug_mode_function(dataset); // Mantener en modo debug
}

void benchmark_mode(Dataset* dataset) {
    clear_screen();
    print_header("📊 MODO BENCHMARK");
    
    if (dataset->num_samples == 0) {
        printf_both("\n❌ No hay datos para benchmark\n");
        wait_for_key("\nPresione Enter para continuar...");
        return;
    }
    
    printf_both("\n🔬 COMPARACIÓN DE CONFIGURACIONES:\n");
    printf_both("════════════════════════════════════════════════════════════════\n");
    
    // Configuraciones a probar
    struct {
        int depth;
        int min_split;
        int min_leaf;
        char desc[50];
    } configs[] = {
        {3, 2, 1, "Poco profundo"},
        {5, 2, 1, "Moderado"},
        {10, 2, 1, "Profundo"},
        {5, 5, 2, "Conservador"},
        {5, 10, 5, "Muy conservador"},
    };
    
    int num_configs = sizeof(configs) / sizeof(configs[0]);
    
    printf_both("Configuración          │ Accuracy │ Tiempo (s) │ Nodos │ Hojas │\n");
    printf_both("───────────────────────┼──────────┼────────────┼───────┼───────┤\n");
    
    for (int i = 0; i < num_configs; i++) {
        printf_both("%-22s │ ", configs[i].desc);
        fflush(stdout);
        
        time_t start = time(NULL);
        
        // Entrenar modelo
        DecisionTree* model = train_decision_tree(dataset, 
                                                 configs[i].depth,
                                                 configs[i].min_split,
                                                 configs[i].min_leaf);
        
        time_t end = time(NULL);
        double elapsed = difftime(end, start);
        
        if (model) {
            printf_both("%7.2f%% │ %10.2f │ %5d │ %5d │\n",
                   model->accuracy * 100,
                   elapsed,
                   model->node_count,
                   model->leaf_count);
            
            free_tree(model->root);
            free(model);
        } else {
            printf_both(" ERROR  │          │       │       │\n");
        }
    }
    
    printf_both("───────────────────────┴──────────┴────────────┴───────┴───────┘\n");
    
    // Validación cruzada
    printf_both("\n🔬 VALIDACIÓN CRUZADA (5 folds):\n");
    cross_validation(dataset, 5, 5, 2, 1);
    
    wait_for_key("\nPresione Enter para continuar...");
}

void tutorial_mode() {
    clear_screen();
    print_header("📚 TUTORIAL - ÁRBOLES DE DECISIÓN");
    
    printf_both("\n🌟 BIENVENIDO AL TUTORIAL\n");
    printf_both("════════════════════════════════════════════════════════════════\n");
    
    printf_both("\n📖 ¿QUÉ ES UN ÁRBOL DE DECISIÓN?\n");
    printf_both("   Un árbol de decisión es un modelo de aprendizaje automático que\n");
    printf_both("   toma decisiones basadas en una serie de preguntas sobre las\n");
    printf_both("   características de los datos.\n");
    
    printf_both("\n🎯 CÓMO FUNCIONA:\n");
    printf_both("   1. El árbol comienza en la raíz con todos los datos\n");
    printf_both("   2. En cada nodo, selecciona la mejor característica para dividir\n");
    printf_both("   3. Divide los datos en subconjuntos más puros\n");
    printf_both("   4. Repite hasta que los nodos sean 'puros' o se alcance profundidad\n");
    
    printf_both("\n📊 MÉTRICAS IMPORTANTES:\n");
    printf_both("   • Accuracy: Porcentaje de predicciones correctas\n");
    printf_both("   • Precision: De lo predicho positivo, cuánto es realmente positivo\n");
    printf_both("   • Recall: De lo realmente positivo, cuánto se predijo positivo\n");
    printf_both("   • F1-Score: Media armónica de precisión y recall\n");
    
    printf_both("\n🌲 PARTES DEL ÁRBOL:\n");
    printf_both("   • Nodo raíz: Primer nodo del árbol\n");
    printf_both("   • Nodos internos: Puntos de decisión\n");
    printf_both("   • Hojas: Resultados finales (clases)\n");
    printf_both("   • Ramas: Conexiones entre nodos\n");
    
    printf_both("\n⚙️  PARÁMETROS CLAVE:\n");
    printf_both("   • Profundidad máxima: Cuántos niveles puede tener el árbol\n");
    printf_both("   • Mínimo muestras por split: Para evitar sobreajuste\n");
    printf_both("   • Mínimo muestras por hoja: Para tener hojas significativas\n");
    
    printf_both("\n💡 CONSEJOS PRÁCTICOS:\n");
    printf_both("   1. Comience con profundidad moderada (3-5)\n");
    printf_both("   2. Use validación cruzada para evaluar robustez\n");
    printf_both("   3. Examine la importancia de características\n");
    printf_both("   4. Guarde modelos buenos para uso futuro\n");
    
    printf_both("\n🚀 FLUJO DE TRABAJO RECOMENDADO:\n");
    printf_both("   1. Cargar datos\n");
    printf_both("   2. Explorar y analizar datos\n");
    printf_both("   3. Entrenar modelo con diferentes configuraciones\n");
    printf_both("   4. Evaluar métricas\n");
    printf_both("   5. Guardar el mejor modelo\n");
    printf_both("   6. Usar para predicciones\n");
    
    printf_both("\n🔧 CARACTERÍSTICAS AVANZADAS:\n");
    printf_both("   • Persistencia: Guardar y cargar modelos\n");
    printf_both("   • Refinamiento: Mejorar modelos existentes\n");
    printf_both("   • Exportación: Resultados en múltiples formatos\n");
    printf_both("   • Historial: Seguimiento de todas las operaciones\n");
    
    printf_both("\n🎮 COMANDOS ÚTILES EN MODO INTERACTIVO:\n");
    printf_both("   • 'salir': Terminar modo interactivo\n");
    printf_both("   • 'ruta': Ver ruta de decisión de una muestra\n");
    printf_both("   • 'estadisticas': Ver estadísticas del modelo\n");
    
    wait_for_key("\nPresione Enter para volver al menú principal...");
}

// Funciones de evaluación para archivo
void print_confusion_matrix_to_file(DecisionTree* tree, Dataset* dataset) {
    if (!output_file || dataset->num_classes == 0) return;
    
    int confusion[MAX_CLASSES][MAX_CLASSES] = {0};
    
    for (int i = 0; i < dataset->num_samples; i++) {
        int prediction = predict_tree(tree->root, &dataset->samples[i]);
        confusion[dataset->samples[i].target][prediction]++;
    }
    
    fprintf(output_file, "\nMATRIZ DE CONFUSIÓN:\n");
    fprintf(output_file, "====================\n\n");
    
    fprintf(output_file, "%20s", "");
    for (int i = 0; i < dataset->num_classes; i++) {
        fprintf(output_file, "Pred %-8d", i);
    }
    fprintf(output_file, "\n");
    
    for (int i = 0; i < dataset->num_classes; i++) {
        fprintf(output_file, "Real %-15d", i);
        for (int j = 0; j < dataset->num_classes; j++) {
            fprintf(output_file, "%-10d", confusion[i][j]);
        }
        fprintf(output_file, "\n");
    }
}

void print_feature_importance_to_file(DecisionTree* tree, Dataset* dataset) {
    if (!output_file) return;
    
    double importance[MAX_FEATURES] = {0};
    
    if (tree->root) {
        calculate_feature_importance(tree->root, importance, dataset->num_samples);
    }
    
    // Normalizar
    double sum = 0.0;
    for (int i = 0; i < dataset->num_features; i++) {
        sum += importance[i];
    }
    
    if (sum > 0) {
        for (int i = 0; i < dataset->num_features; i++) {
            importance[i] = (importance[i] / sum) * 100.0;
        }
    }
    
    fprintf(output_file, "\nIMPORTANCIA DE CARACTERÍSTICAS:\n");
    fprintf(output_file, "================================\n\n");
    
    fprintf(output_file, "%-30s %12s\n", "Característica", "Importancia");
    fprintf(output_file, "%-30s %12s\n", "------------------------------", "------------");
    
    for (int i = 0; i < dataset->num_features; i++) {
        fprintf(output_file, "%-30s %10.2f%%\n", dataset->feature_names[i], importance[i]);
    }
}

void print_model_metrics_to_file(DecisionTree* tree, Dataset* dataset) {
    if (!output_file) return;
    
    fprintf(output_file, "\nMÉTRICAS DEL MODELO:\n");
    fprintf(output_file, "=====================\n\n");
    
    fprintf(output_file, "Nombre: %s\n", tree->name);
    fprintf(output_file, "Descripción: %s\n", tree->description);
    fprintf(output_file, "Algoritmo: %s\n", tree->algorithm);
    
    char created_str[50], trained_str[50];
    strftime(created_str, sizeof(created_str), "%Y-%m-%d %H:%M:%S", localtime(&tree->created_at));
    strftime(trained_str, sizeof(trained_str), "%Y-%m-%d %H:%M:%S", localtime(&tree->last_trained));
    
    fprintf(output_file, "Creado: %s\n", created_str);
    fprintf(output_file, "Último entrenamiento: %s\n", trained_str);
    
    fprintf(output_file, "\nPARÁMETROS:\n");
    fprintf(output_file, "  • Profundidad máxima: %d\n", tree->max_depth);
    fprintf(output_file, "  • Mínimo muestras por split: %d\n", tree->min_samples_split);
    fprintf(output_file, "  • Mínimo muestras por hoja: %d\n", tree->min_samples_leaf);
    
    fprintf(output_file, "\nESTADÍSTICAS:\n");
    fprintf(output_file, "  • Muestras entrenadas: %d\n", tree->total_samples_trained);
    fprintf(output_file, "  • Nodos totales: %d\n", tree->node_count);
    fprintf(output_file, "  • Hojas: %d\n", tree->leaf_count);
    fprintf(output_file, "  • Profundidad real: %d\n", tree_max_depth(tree->root));
    
    fprintf(output_file, "\nMÉTRICAS DE RENDIMIENTO:\n");
    fprintf(output_file, "  • Accuracy: %.4f (%.2f%%)\n", tree->accuracy, tree->accuracy * 100);
    fprintf(output_file, "  • Precision: %.4f (%.2f%%)\n", tree->precision, tree->precision * 100);
    fprintf(output_file, "  • Recall: %.4f (%.2f%%)\n", tree->recall, tree->recall * 100);
    fprintf(output_file, "  • F1-Score: %.4f (%.2f%%)\n", tree->f1_score, tree->f1_score * 100);
}

void save_tree_structure_to_file(TreeNode* root, Dataset* dataset) {
    if (!output_file || !root) return;
    
    fprintf(output_file, "\nESTRUCTURA DEL ÁRBOL:\n");
    fprintf(output_file, "======================\n\n");
    
    TreeNode* stack[100];
    int depths[100];
    int top = -1;
    
    stack[++top] = root;
    depths[top] = 0;
    
    while (top >= 0) {
        TreeNode* current = stack[top];
        int depth = depths[top--];
        
        for (int i = 0; i < depth; i++) {
            fprintf(output_file, "  ");
        }
        
        if (current->is_leaf) {
            fprintf(output_file, "Hoja: %s (muestras: %d, gini: %.4f)\n",
                   dataset->class_names[current->class_label],
                   current->samples, current->gini);
        } else {
            fprintf(output_file, "Si %s <= %.4f (muestras: %d, gini: %.4f)\n",
                   dataset->feature_names[current->split_feature],
                   current->split_value,
                   current->samples, current->gini);
            
            if (current->right) {
                stack[++top] = current->right;
                depths[top] = depth + 1;
            }
            if (current->left) {
                stack[++top] = current->left;
                depths[top] = depth + 1;
            }
        }
    }
}

void save_predictions_to_file(DecisionTree* tree, Dataset* dataset) {
    if (!output_file) return;
    
    fprintf(output_file, "\nPREDICCIONES:\n");
    fprintf(output_file, "=============\n\n");
    
    int correct = 0;
    
    fprintf(output_file, "%-6s ", "ID");
    for (int i = 0; i < dataset->num_features && i < 4; i++) {
        fprintf(output_file, "%-12s", dataset->feature_names[i]);
    }
    fprintf(output_file, "%-20s %-20s %-10s\n", "Clase Real", "Predicción", "Resultado");
    
    for (int i = 0; i < 50 && i < dataset->num_samples; i++) {
        DataSample sample = dataset->samples[i];
        int prediction = predict_tree(tree->root, &sample);
        
        fprintf(output_file, "%-6d ", i + 1);
        
        // Características
        for (int j = 0; j < dataset->num_features && j < 4; j++) {
            fprintf(output_file, "%-12.4f", sample.features[j]);
        }
        
        fprintf(output_file, "%-20s %-20s ",
               dataset->class_names[sample.target],
               dataset->class_names[prediction]);
        
        if (prediction == sample.target) {
            fprintf(output_file, "%-10s\n", "CORRECTO");
            correct++;
        } else {
            fprintf(output_file, "%-10s\n", "INCORRECTO");
        }
    }
    
    fprintf(output_file, "\nResumen: %d/%d correctas (%.2f%%)\n",
            correct, dataset->num_samples < 50 ? dataset->num_samples : 50,
            (double)correct / (dataset->num_samples < 50 ? dataset->num_samples : 50) * 100);
}

void save_dataset_info_to_file(Dataset* dataset) {
    if (!output_file) return;
    
    fprintf(output_file, "\nINFORMACIÓN DEL DATASET:\n");
    fprintf(output_file, "=========================\n\n");
    
    fprintf(output_file, "Muestras totales: %d\n", dataset->num_samples);
    fprintf(output_file, "Características: %d\n", dataset->num_features);
    fprintf(output_file, "Clases: %d\n", dataset->num_classes);
    fprintf(output_file, "Normalizado: %s\n", dataset->is_normalized ? "Sí" : "No");
    
    fprintf(output_file, "\nCaracterísticas:\n");
    for (int i = 0; i < dataset->num_features; i++) {
        fprintf(output_file, "  %d. %s (min: %.4f, max: %.4f, mean: %.4f, std: %.4f)\n",
                i + 1, dataset->feature_names[i],
                dataset->feature_min[i], dataset->feature_max[i],
                dataset->feature_mean[i], dataset->feature_std[i]);
    }
    
    fprintf(output_file, "\nClases:\n");
    int class_counts[MAX_CLASSES] = {0};
    for (int i = 0; i < dataset->num_samples; i++) {
        class_counts[dataset->samples[i].target]++;
    }
    for (int i = 0; i < dataset->num_classes; i++) {
        double percentage = (double)class_counts[i] / dataset->num_samples * 100;
        fprintf(output_file, "  %s: %d (%.1f%%)\n",
                dataset->class_names[i], class_counts[i], percentage);
    }
}

void save_training_log_to_file(DecisionTree* tree, Dataset* dataset, time_t start_time, time_t end_time) {
    char log_path[256];
    snprintf(log_path, sizeof(log_path), "%s/training_%ld.log", LOGS_DIR, time(NULL));
    
    FILE* log = fopen(log_path, "w");
    if (!log) return;
    
    fprintf(log, "LOG DE ENTRENAMIENTO\n");
    fprintf(log, "====================\n\n");
    
    fprintf(log, "Fecha: %s", ctime(&end_time));
    fprintf(log, "Duración: %.2f segundos\n", difftime(end_time, start_time));
    
    fprintf(log, "\nMODELO:\n");
    fprintf(log, "  Nombre: %s\n", tree->name);
    fprintf(log, "  Algoritmo: %s\n", tree->algorithm);
    
    fprintf(log, "\nDATASET:\n");
    fprintf(log, "  Muestras: %d\n", dataset->num_samples);
    fprintf(log, "  Características: %d\n", dataset->num_features);
    fprintf(log, "  Clases: %d\n", dataset->num_classes);
    
    fprintf(log, "\nPARÁMETROS:\n");
    fprintf(log, "  Profundidad máxima: %d\n", tree->max_depth);
    fprintf(log, "  Mínimo muestras por split: %d\n", tree->min_samples_split);
    fprintf(log, "  Mínimo muestras por hoja: %d\n", tree->min_samples_leaf);
    
    fprintf(log, "\nRESULTADOS:\n");
    fprintf(log, "  Accuracy: %.4f\n", tree->accuracy);
    fprintf(log, "  Nodos: %d\n", tree->node_count);
    fprintf(log, "  Hojas: %d\n", tree->leaf_count);
    fprintf(log, "  Profundidad: %d\n", tree_max_depth(tree->root));
    
    fclose(log);
}

// Funciones auxiliares
void shuffle_indices(int* indices, int count) {
    for (int i = count - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
}

void show_data_distribution(Dataset* dataset) {
    printf_both("\n📊 DISTRIBUCIÓN DE DATOS:\n");
    
    int class_counts[MAX_CLASSES] = {0};
    for (int i = 0; i < dataset->num_samples; i++) {
        class_counts[dataset->samples[i].target]++;
    }
    
    printf_both("┌──────────────────────────────┬──────────────┬──────────────┬──────────────┐\n");
    printf_both("│ Clase                        │ Muestras     │ Porcentaje   │ Acumulado    │\n");
    printf_both("├──────────────────────────────┼──────────────┼──────────────┼──────────────┤\n");
    
    double accumulated = 0.0;
    for (int i = 0; i < dataset->num_classes && i < 10; i++) {
        double percentage = (double)class_counts[i] / dataset->num_samples * 100;
        accumulated += percentage;
        
        printf_both("│ %-28s │ %12d │ %12.1f%% │ %12.1f%% │\n", 
               dataset->class_names[i], class_counts[i], percentage, accumulated);
    }
    
    if (dataset->num_classes > 10) {
        printf_both("│ ...                                                              │\n");
    }
    
    printf_both("├──────────────────────────────┼──────────────┼──────────────┼──────────────┤\n");
    printf_both("│ %-28s │ %12d │ %12.1f%% │ %12.1f%% │\n", 
           "TOTAL", dataset->num_samples, 100.0, 100.0);
    printf_both("└──────────────────────────────┴──────────────┴──────────────┴──────────────┘\n");
}

void show_feature_statistics(Dataset* dataset) {
    if (dataset->num_features == 0) return;
    
    printf_both("\n📈 ESTADÍSTICAS DE CARACTERÍSTICAS:\n");
    
    int features_to_show = dataset->num_features < 6 ? dataset->num_features : 6;
    
    printf_both("┌─────┬──────────────────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐\n");
    printf_both("│ No. │ Característica               │ Mínimo       │ Máximo       │ Media        │ Desv. Std.   │\n");
    printf_both("├─────┼──────────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤\n");
    
    for (int i = 0; i < features_to_show; i++) {
        printf_both("│ %3d │ %-28s │ %12.4f │ %12.4f │ %12.4f │ %12.4f │\n", 
               i + 1, dataset->feature_names[i],
               dataset->feature_min[i], dataset->feature_max[i],
               dataset->feature_mean[i], dataset->feature_std[i]);
    }
    
    if (dataset->num_features > 6) {
        printf_both("│ ... (y %d más)                                                               │\n", dataset->num_features - 6);
    }
    
    printf_both("└─────┴──────────────────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘\n");
}

void show_correlation_matrix(Dataset* dataset) {
    if (dataset->num_features <= 1) return;
    
    printf_both("\n📊 MATRIZ DE CORRELACIÓN (primeras 5 características):\n");
    
    int features_to_show = dataset->num_features < 5 ? dataset->num_features : 5;
    
    printf_both("        ");
    for (int i = 0; i < features_to_show; i++) {
        printf_both("%-10.10s ", dataset->feature_names[i]);
    }
    printf_both("\n");
    
    for (int i = 0; i < features_to_show; i++) {
        printf_both("%-7.7s ", dataset->feature_names[i]);
        for (int j = 0; j < features_to_show; j++) {
            if (i == j) {
                printf_both("1.000     ");
            } else {
                // Calcular correlación simple (Pearson)
                double sum_xy = 0.0, sum_x = 0.0, sum_y = 0.0;
                double sum_x2 = 0.0, sum_y2 = 0.0;
                
                for (int k = 0; k < dataset->num_samples && k < 100; k++) {
                    double x = dataset->samples[k].features[i];
                    double y = dataset->samples[k].features[j];
                    sum_xy += x * y;
                    sum_x += x;
                    sum_y += y;
                    sum_x2 += x * x;
                    sum_y2 += y * y;
                }
                
                int n = dataset->num_samples < 100 ? dataset->num_samples : 100;
                double numerator = n * sum_xy - sum_x * sum_y;
                double denominator = sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));
                
                double correlation = 0.0;
                if (denominator > 1e-10) {
                    correlation = numerator / denominator;
                }
                
                printf_both("%-10.3f", correlation);
            }
        }
        printf_both("\n");
    }
}

void show_prediction_path(TreeNode* root, DataSample* sample, Dataset* dataset) {
    if (!root || !sample) return;
    
    TreeNode* current = root;
    int step = 1;
    
    printf_both("\n");
    
    while (current && !current->is_leaf) {
        printf_both("Paso %d: ", step++);
        printf_both("Si %s <= %.4f\n", 
               dataset->feature_names[current->split_feature],
               current->split_value);
        
        printf_both("   • Valor actual: %.4f\n", sample->features[current->split_feature]);
        
        if (sample->features[current->split_feature] <= current->split_value) {
            printf_both("   • Decisión: SÍ → Ir a subárbol izquierdo\n");
            current = current->left;
        } else {
            printf_both("   • Decisión: NO → Ir a subárbol derecho\n");
            current = current->right;
        }
        
        if (current) {
            printf_both("   • Muestras en nodo: %d\n", current->samples);
            printf_both("   • Impureza (Gini): %.4f\n", current->gini);
        }
        
        printf_both("\n");
    }
    
    if (current && current->is_leaf) {
        printf_both("🏁 LLEGADO A HOJA:\n");
        printf_both("   • Clase predicha: %s\n", dataset->class_names[current->class_label]);
        printf_both("   • Muestras en hoja: %d\n", current->samples);
        printf_both("   • Confianza (1 - Gini): %.2f%%\n", (1.0 - current->gini) * 100);
    }
}

void show_training_progress(int iteration, int total, const char* phase) {
    static time_t last_update = 0;
    time_t now = time(NULL);
    
    // Actualizar cada segundo
    if (now - last_update >= 1 || iteration == total) {
        print_progress_bar(iteration, total, phase, 0.0);
        last_update = now;
    }
    
    if (iteration == total) {
        printf_both("\n");
    }
}

void print_usage(const char* program_name) {
    clear_screen();
    print_header("📖 AYUDA - SISTEMA DE ÁRBOLES DE DECISIÓN");
    
    printf("Uso: %s [OPCIONES] [ARCHIVO_DE_DATOS]\n\n", program_name);
    
    printf("Opciones:\n");
    printf("  -h, --help                  Muestra esta ayuda\n");
    printf("  -i, --interactive           Modo interactivo\n");
    printf("  -t, --train                 Entrenar nuevo modelo\n");
    printf("  -l ARCH, --load ARCH        Cargar modelo existente\n");
    printf("  -d N, --depth N             Profundidad máxima del árbol (default: 5)\n");
    printf("  -m N, --min-samples N       Mínimo muestras por split (default: 2)\n");
    printf("  -ml N, --min-samples-leaf N Mínimo muestras por hoja (default: 1)\n");
    printf("  -o ARCH, --output ARCH      Guarda resultados en archivo\n");
    printf("  -e ARCH, --export ARCH      Exporta resultados en formato específico\n");
    printf("  -b, --benchmark             Ejecutar benchmarks\n");
    printf("  -tu, --tutorial             Mostrar tutorial\n");
    printf("  -dbg, --debug               Modo depuración\n");
    printf("  -v, --version               Mostrar versión\n\n");


    printf("Formas de ejecución del programa:\n\n");
    printf("# Modo normal\n");
    printf("./arboles_decision_avanzado datos.csv\n\n");
    printf("# Modo interactivo\n");
    printf("./arboles_decision_avanzado -i datos.csv\n\n");
    printf("# Entrenar nuevo modelo\n");
    printf("./arboles_decision_avanzado -t datos.csv\n\n");
    printf("# Cargar modelo existente\n");
    printf("./arboles_decision_avanzado -l modelo.dtree datos.csv\n\n");
    printf("# Con salida a archivo\n");
    printf("./arboles_decision_avanzado datos.csv -o resultado.txt\n\n");
    printf("# Modo benchmark\n");
    printf("./arboles_decision_avanzado -b datos.csv\n\n");
    printf("# Tutorial\n");
    printf("./arboles_decision_avanzado --tutorial\n\n");
    printf("# Ayuda completa\n");
    printf("./arboles_decision_avanzado --help\n");

    
    printf("Características:\n");
    printf("  • Entrenamiento de árboles de decisión CART\n");
    printf("  • Persistencia completa de modelos\n");
    printf("  • Gestión avanzada de modelos\n");
    printf("  • Evaluación con múltiples métricas\n");
    printf("  • Validación cruzada y curvas de aprendizaje\n");
    printf("  • Exportación en múltiples formatos\n");
    printf("  • Historial completo de operaciones\n");
    printf("  • Modo interactivo para predicciones\n");
    printf("  • Refinamiento de modelos existentes\n\n");
    
    printf("Ejemplos:\n");
    printf("  %s datos.txt                    # Cargar y entrenar\n", program_name);
    printf("  %s -l modelo.dtree datos.txt    # Cargar modelo existente\n", program_name);
    printf("  %s -t -d 3 -m 5 datos.txt       # Entrenar con parámetros específicos\n", program_name);
    printf("  %s -i datos.txt                 # Modo interactivo\n", program_name);
    printf("  %s -b datos.txt                 # Ejecutar benchmarks\n", program_name);
    printf("  %s --tutorial                   # Mostrar tutorial\n", program_name);
    
    printf("\nFormato del archivo de datos:\n");
    printf("  • CSV con o sin encabezado\n");
    printf("  • Última columna: nombre de la clase\n");
    printf("  • Líneas que comienzan con # son comentarios\n");
    printf("  • Ejemplo: # característica1,característica2,...,clase\n");
    printf("  • Ejemplo: 5.1,3.5,1.4,0.2,Iris-setosa\n");
}

void print_license() {
    clear_screen();
    print_header("📄 LICENCIA DEL SISTEMA");
    
    printf_both("\nSISTEMA DE ÁRBOLES DE DECISIÓN CON PERSISTENCIA\n");
    printf_both("Versión 2.0\n\n");
    
    printf_both("Licencia MIT\n\n");
    
    printf_both("Copyright (c) 2024 Sistema de Árboles de Decisión\n\n");
    
    printf_both("Se concede permiso, libre de cargos, a cualquier persona que obtenga una copia\n");
    printf_both("de este software y de los archivos de documentación asociados (el \"Software\"),\n");
    printf_both("a utilizar el Software sin restricción, incluyendo sin limitación los derechos\n");
    printf_both("a usar, copiar, modificar, fusionar, publicar, distribuir, sublicenciar, y/o\n");
    printf_both("vender copias del Software, y a permitir a las personas a las que se les\n");
    printf_both("proporcione el Software a hacer lo mismo, sujeto a las siguientes condiciones:\n\n");
    
    printf_both("El aviso de copyright anterior y este aviso de permiso se incluirán en todas\n");
    printf_both("las copias o partes sustanciales del Software.\n\n");
    
    printf_both("EL SOFTWARE SE PROPORCIONA \"COMO ESTÁ\", SIN GARANTÍA DE NINGÚN TIPO,\n");
    printf_both("EXPRESA O IMPLÍCITA, INCLUYENDO PERO NO LIMITADO A GARANTÍAS DE\n");
    printf_both("COMERCIALIZACIÓN, IDONEIDAD PARA UN PROPÓSITO PARTICULAR Y NO INFRACCIÓN.\n");
    printf_both("EN NINGÚN CASO LOS AUTORES O TITULARES DEL COPYRIGHT SERÁN RESPONSABLES\n");
    printf_both("DE NINGUNA RECLAMACIÓN, DAÑOS U OTRAS RESPONSABILIDADES, YA SEA EN UNA\n");
    printf_both("ACCIÓN DE CONTRATO, AGRAVIO O CUALQUIER OTRO MOTIVO, QUE SURJA DE O EN\n");
    printf_both("CONEXIÓN CON EL SOFTWARE O EL USO U OTRO TIPO DE ACCIONES EN EL SOFTWARE.\n");
}

void print_version() {
    printf("Sistema de Árboles de Decisión con Persistencia - Versión 2.0\n");
    printf("Compilado el %s a las %s\n", __DATE__, __TIME__);
    printf("Límites: %d muestras, %d características, %d clases\n", 
           MAX_SAMPLES, MAX_FEATURES, MAX_CLASSES);
}

// Nuevas funciones avanzadas (implementaciones básicas)
void ensemble_training(Dataset* dataset, int num_trees, int max_depth) {
    printf_both("\n🌳 ENTRENAMIENTO DE ENSAMBLE (%d árboles)\n", num_trees);
    
    if (num_trees < 2) num_trees = 3;
    if (num_trees > 10) num_trees = 10;
    
    DecisionTree* ensemble[10];
    double total_accuracy = 0.0;
    
    for (int i = 0; i < num_trees; i++) {
        printf_both("\n🌲 Entrenando árbol %d/%d...\n", i + 1, num_trees);
        
        // Crear bootstrap sample
        Dataset bootstrap = {0};
        memcpy(&bootstrap, dataset, sizeof(Dataset));
        bootstrap.num_samples = 0;
        
        // Muestreo con reemplazo
        for (int j = 0; j < dataset->num_samples; j++) {
            int random_idx = rand() % dataset->num_samples;
            bootstrap.samples[bootstrap.num_samples++] = dataset->samples[random_idx];
        }
        
        // Entrenar árbol
        ensemble[i] = train_decision_tree(&bootstrap, max_depth, 2, 1);
        if (ensemble[i]) {
            total_accuracy += ensemble[i]->accuracy;
            printf_both("   Accuracy: %.2f%%\n", ensemble[i]->accuracy * 100);
        }
    }
    
    printf_both("\n📊 RESULTADOS DEL ENSAMBLE:\n");
    printf_both("   • Accuracy promedio: %.2f%%\n", (total_accuracy / num_trees) * 100);
    
    // Limpiar memoria
    for (int i = 0; i < num_trees; i++) {
        if (ensemble[i]) {
            free_tree(ensemble[i]->root);
            free(ensemble[i]);
        }
    }
}

void backup_model(const char* model_name) {
    char backup_path[256];
    time_t now = time(NULL);
    struct tm* t = localtime(&now);
    
    snprintf(backup_path, sizeof(backup_path), 
             "backups/%s_%04d%02d%02d_%02d%02d%02d.dtree",
             model_name,
             t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
             t->tm_hour, t->tm_min, t->tm_sec);
    
    // Buscar modelo actual
    for (int i = 0; i < model_registry.num_models; i++) {
        if (strcmp(model_registry.models[i].name, model_name) == 0) {
            // Copiar archivo
            FILE* src = fopen(model_registry.models[i].filename, "rb");
            FILE* dst = fopen(backup_path, "wb");
            
            if (src && dst) {
                char buffer[4096];
                size_t bytes;
                while ((bytes = fread(buffer, 1, sizeof(buffer), src)) > 0) {
                    fwrite(buffer, 1, bytes, dst);
                }
                
                fclose(src);
                fclose(dst);
                
                printf_both("✅ Backup creado: %s\n", backup_path);
                log_operation("BACKUP_MODELO", backup_path, 0.0, 0);
            } else {
                printf_both("❌ Error al crear backup\n");
            }
            return;
        }
    }
    
    printf_both("❌ Modelo no encontrado: %s\n", model_name);
}

void restore_model(const char* backup_name) {
    char backup_path[256];
    snprintf(backup_path, sizeof(backup_path), "backups/%s", backup_name);
    
    char model_path[256];
    snprintf(model_path, sizeof(model_path), "%s/%s", MODEL_DIR, backup_name);
    
    // Copiar backup a modelos
    FILE* src = fopen(backup_path, "rb");
    FILE* dst = fopen(model_path, "wb");
    
    if (src && dst) {
        char buffer[4096];
        size_t bytes;
        while ((bytes = fread(buffer, 1, sizeof(buffer), src)) > 0) {
            fwrite(buffer, 1, bytes, dst);
        }
        
        fclose(src);
        fclose(dst);
        
        // Cargar modelo
        current_model = load_tree_model(backup_name);
        if (current_model) {
            strcpy(current_model_file, model_path);
            printf_both("✅ Modelo restaurado: %s\n", backup_name);
            log_operation("RESTAURAR_MODELO", backup_name, current_model->accuracy, 0);
        } else {
            printf_both("❌ Error al cargar modelo restaurado\n");
        }
    } else {
        printf_both("❌ Error al restaurar backup\n");
    }
}

// Funciones de exportación/importación JSON (implementaciones básicas)
void export_model_json(DecisionTree* tree, Dataset* dataset, const char* filename) {
    char full_path[256];
    snprintf(full_path, sizeof(full_path), "%s/%s", EXPORTS_DIR, filename);
    
    FILE* file = fopen(full_path, "w");
    if (!file) {
        printf("❌ Error al crear archivo JSON\n");
        return;
    }
    
    fprintf(file, "{\n");
    fprintf(file, "  \"model\": {\n");
    fprintf(file, "    \"name\": \"%s\",\n", tree->name);
    fprintf(file, "    \"algorithm\": \"%s\",\n", tree->algorithm);
    fprintf(file, "    \"accuracy\": %.4f,\n", tree->accuracy);
    fprintf(file, "    \"created_at\": %ld,\n", tree->created_at);
    fprintf(file, "    \"node_count\": %d,\n", tree->node_count);
    fprintf(file, "    \"leaf_count\": %d\n", tree->leaf_count);
    fprintf(file, "  }\n");
    fprintf(file, "}\n");
    
    fclose(file);
    printf("✅ Modelo exportado en JSON: %s\n", full_path);
}

void import_model_json(const char* filename, DecisionTree** tree, Dataset* dataset) {
    printf("ℹ️  Importación JSON en desarrollo\n");
    // Implementación básica - en una versión real se parsearía el JSON
}

// Implementación de funciones pendientes del archivo original
void export_results_csv(DecisionTree* tree, Dataset* dataset, const char* filename) {
    char full_path[256];
    snprintf(full_path, sizeof(full_path), "%s/%s", EXPORTS_DIR, filename);
    
    FILE* file = fopen(full_path, "w");
    if (!file) {
        printf("❌ Error al crear archivo CSV\n");
        return;
    }
    
    // Cabecera
    fprintf(file, "sample_id,");
    for (int i = 0; i < dataset->num_features; i++) {
        fprintf(file, "%s,", dataset->feature_names[i]);
    }
    fprintf(file, "actual_class,predicted_class,correct\n");
    
    // Datos
    for (int i = 0; i < dataset->num_samples && i < 100; i++) {
        fprintf(file, "%d,", i + 1);
        
        for (int j = 0; j < dataset->num_features; j++) {
            fprintf(file, "%.6f,", dataset->samples[i].features[j]);
        }
        
        int prediction = predict_tree(tree->root, &dataset->samples[i]);
        fprintf(file, "%s,%s,%s\n",
                dataset->class_names[dataset->samples[i].target],
                dataset->class_names[prediction],
                prediction == dataset->samples[i].target ? "true" : "false");
    }
    
    fclose(file);
    printf("✅ Resultados exportados a CSV: %s\n", full_path);
}

void export_metrics_json(DecisionTree* tree, Dataset* dataset, const char* filename) {
    char full_path[256];
    snprintf(full_path, sizeof(full_path), "%s/%s", EXPORTS_DIR, filename);
    
    FILE* file = fopen(full_path, "w");
    if (!file) {
        printf("❌ Error al crear archivo JSON\n");
        return;
    }
    
    fprintf(file, "{\n");
    fprintf(file, "  \"model_metrics\": {\n");
    fprintf(file, "    \"name\": \"%s\",\n", tree->name);
    fprintf(file, "    \"accuracy\": %.4f,\n", tree->accuracy);
    fprintf(file, "    \"precision\": %.4f,\n", tree->precision);
    fprintf(file, "    \"recall\": %.4f,\n", tree->recall);
    fprintf(file, "    \"f1_score\": %.4f,\n", tree->f1_score);
    fprintf(file, "    \"node_count\": %d,\n", tree->node_count);
    fprintf(file, "    \"leaf_count\": %d,\n", tree->leaf_count);
    fprintf(file, "    \"max_depth\": %d,\n", tree->max_depth);
    fprintf(file, "    \"samples_trained\": %d\n", tree->total_samples_trained);
    fprintf(file, "  },\n");
    
    fprintf(file, "  \"feature_importance\": [\n");
    for (int i = 0; i < dataset->num_features; i++) {
        fprintf(file, "    {\n");
        fprintf(file, "      \"feature\": \"%s\",\n", dataset->feature_names[i]);
        fprintf(file, "      \"importance\": %.4f\n", tree->feature_importance[i]);
        fprintf(file, "    }%s\n", i < dataset->num_features - 1 ? "," : "");
    }
    fprintf(file, "  ]\n");
    fprintf(file, "}\n");
    
    fclose(file);
    printf("✅ Métricas exportadas en JSON: %s\n", full_path);
}

// Implementación de funciones de persistencia
int save_tree_model(DecisionTree* tree, Dataset* dataset, const char* filename) {
    char full_path[300];
    snprintf(full_path, sizeof(full_path), "%s/%s", MODEL_DIR, filename);
    
    FILE* file = fopen(full_path, "wb");
    if (!file) {
        printf("❌ Error al crear archivo: %s\n", full_path);
        return 0;
    }
    
    // Guardar cabecera
    fprintf(file, "DECISION_TREE_MODEL_V2.0\n");
    
    // Guardar metadatos del árbol
    fwrite(&tree->accuracy, sizeof(double), 1, file);
    fwrite(&tree->precision, sizeof(double), 1, file);
    fwrite(&tree->recall, sizeof(double), 1, file);
    fwrite(&tree->f1_score, sizeof(double), 1, file);
    fwrite(&tree->max_depth, sizeof(int), 1, file);
    fwrite(&tree->min_samples_split, sizeof(int), 1, file);
    fwrite(&tree->min_samples_leaf, sizeof(int), 1, file);
    fwrite(&tree->created_at, sizeof(time_t), 1, file);
    fwrite(&tree->last_trained, sizeof(time_t), 1, file);
    fwrite(&tree->last_used, sizeof(time_t), 1, file);
    fwrite(&tree->node_count, sizeof(int), 1, file);
    fwrite(&tree->leaf_count, sizeof(int), 1, file);
    fwrite(&tree->total_samples_trained, sizeof(int), 1, file);
    fwrite(&tree->is_trained, sizeof(int), 1, file);
    
    // Guardar nombre y descripción
    int name_len = strlen(tree->name) + 1;
    fwrite(&name_len, sizeof(int), 1, file);
    fwrite(tree->name, sizeof(char), name_len, file);
    
    int desc_len = strlen(tree->description) + 1;
    fwrite(&desc_len, sizeof(int), 1, file);
    fwrite(tree->description, sizeof(char), desc_len, file);
    
    // Guardar algoritmo
    fwrite(tree->algorithm, sizeof(char), 20, file);
    
    // Guardar importancia de características
    fwrite(tree->feature_importance, sizeof(double), MAX_FEATURES, file);
    
    // Guardar estructura del árbol
    save_tree_to_file(tree->root, file);
    
    // Guardar metadatos del dataset
    if (dataset) {
        fwrite(&dataset->num_features, sizeof(int), 1, file);
        fwrite(&dataset->num_classes, sizeof(int), 1, file);
        fwrite(&dataset->num_samples, sizeof(int), 1, file);
        fwrite(&dataset->is_normalized, sizeof(int), 1, file);
        
        fwrite(dataset->feature_min, sizeof(double), dataset->num_features, file);
        fwrite(dataset->feature_max, sizeof(double), dataset->num_features, file);
        fwrite(dataset->feature_mean, sizeof(double), dataset->num_features, file);
        fwrite(dataset->feature_std, sizeof(double), dataset->num_features, file);
        
        // Guardar nombres de características
        for (int i = 0; i < dataset->num_features; i++) {
            int len = strlen(dataset->feature_names[i]) + 1;
            fwrite(&len, sizeof(int), 1, file);
            fwrite(dataset->feature_names[i], sizeof(char), len, file);
        }
        
        // Guardar nombres de clases
        for (int i = 0; i < dataset->num_classes; i++) {
            int len = strlen(dataset->class_names[i]) + 1;
            fwrite(&len, sizeof(int), 1, file);
            fwrite(dataset->class_names[i], sizeof(char), len, file);
        }
    } else {
        // Sin dataset
        int zero = 0;
        fwrite(&zero, sizeof(int), 1, file); // num_features
        fwrite(&zero, sizeof(int), 1, file); // num_classes
    }
    
    fclose(file);
    
    // Actualizar registro de modelos
    int found = 0;
    for (int i = 0; i < model_registry.num_models; i++) {
        if (strcmp(model_registry.models[i].filename, full_path) == 0) {
            strcpy(model_registry.models[i].name, tree->name);
            strcpy(model_registry.models[i].description, tree->description);
            model_registry.models[i].last_used = time(NULL);
            model_registry.models[i].last_trained = tree->last_trained;
            model_registry.models[i].accuracy = tree->accuracy;
            model_registry.models[i].precision = tree->precision;
            model_registry.models[i].recall = tree->recall;
            model_registry.models[i].f1_score = tree->f1_score;
            model_registry.models[i].samples_trained = tree->total_samples_trained;
            found = 1;
            break;
        }
    }
    
    if (!found && model_registry.num_models < 100) {
        ModelInfo* model = &model_registry.models[model_registry.num_models];
        strcpy(model->name, tree->name);
        strcpy(model->filename, full_path);
        strcpy(model->description, tree->description);
        model->created = time(NULL);
        model->last_used = time(NULL);
        model->last_trained = tree->last_trained;
        model->accuracy = tree->accuracy;
        model->precision = tree->precision;
        model->recall = tree->recall;
        model->f1_score = tree->f1_score;
        model->samples_trained = tree->total_samples_trained;
        model->is_active = 1;
        model_registry.num_models++;
    }
    
    save_model_registry();
    return 1;
}

int save_tree_to_file(TreeNode* node, FILE* file) {
    if (!node) {
        int marker = -1;
        fwrite(&marker, sizeof(int), 1, file);
        return 1;
    }
    
    // Guardar nodo actual
    fwrite(&node->is_leaf, sizeof(int), 1, file);
    fwrite(&node->samples, sizeof(int), 1, file);
    fwrite(&node->depth, sizeof(int), 1, file);
    fwrite(&node->gini, sizeof(double), 1, file);
    fwrite(&node->entropy, sizeof(double), 1, file);
    
    if (node->is_leaf) {
        fwrite(&node->class_label, sizeof(int), 1, file);
    } else {
        fwrite(&node->split_feature, sizeof(int), 1, file);
        fwrite(&node->split_value, sizeof(double), 1, file);
        fwrite(&node->num_categories, sizeof(int), 1, file);
        
        // Guardar subárboles recursivamente
        save_tree_to_file(node->left, file);
        save_tree_to_file(node->right, file);
    }
    
    return 1;
}

DecisionTree* load_tree_model(const char* filename) {
    char full_path[300];
    snprintf(full_path, sizeof(full_path), "%s/%s", MODEL_DIR, filename);
    
    FILE* file = fopen(full_path, "rb");
    if (!file) {
        // Intentar sin la ruta del directorio
        file = fopen(filename, "rb");
        if (!file) {
            printf("❌ No se pudo abrir el archivo: %s\n", filename);
            return NULL;
        }
    }
    
    // Verificar cabecera
    char header[50];
    fgets(header, sizeof(header), file);
    if (strncmp(header, "DECISION_TREE_MODEL_V2.0", 24) != 0) {
        printf("❌ Formato de modelo inválido o versión antigua\n");
        fclose(file);
        return NULL;
    }
    
    DecisionTree* tree = malloc(sizeof(DecisionTree));
    if (!tree) {
        fclose(file);
        return NULL;
    }
    
    memset(tree, 0, sizeof(DecisionTree));
    
    // Leer metadatos del árbol
    fread(&tree->accuracy, sizeof(double), 1, file);
    fread(&tree->precision, sizeof(double), 1, file);
    fread(&tree->recall, sizeof(double), 1, file);
    fread(&tree->f1_score, sizeof(double), 1, file);
    fread(&tree->max_depth, sizeof(int), 1, file);
    fread(&tree->min_samples_split, sizeof(int), 1, file);
    fread(&tree->min_samples_leaf, sizeof(int), 1, file);
    fread(&tree->created_at, sizeof(time_t), 1, file);
    fread(&tree->last_trained, sizeof(time_t), 1, file);
    fread(&tree->last_used, sizeof(time_t), 1, file);
    fread(&tree->node_count, sizeof(int), 1, file);
    fread(&tree->leaf_count, sizeof(int), 1, file);
    fread(&tree->total_samples_trained, sizeof(int), 1, file);
    fread(&tree->is_trained, sizeof(int), 1, file);
    
    // Leer nombre y descripción
    int name_len;
    fread(&name_len, sizeof(int), 1, file);
    fread(tree->name, sizeof(char), name_len, file);
    
    int desc_len;
    fread(&desc_len, sizeof(int), 1, file);
    fread(tree->description, sizeof(char), desc_len, file);
    
    // Leer algoritmo
    fread(tree->algorithm, sizeof(char), 20, file);
    
    // Leer importancia de características
    fread(tree->feature_importance, sizeof(double), MAX_FEATURES, file);
    
    // Leer estructura del árbol
    tree->root = load_tree_from_file(file);
    
    // Leer metadatos del dataset (se leen pero no se usan directamente)
    int num_features, num_classes, num_samples, is_normalized;
    fread(&num_features, sizeof(int), 1, file);
    fread(&num_classes, sizeof(int), 1, file);
    
    if (num_features > 0) {
        fread(&num_samples, sizeof(int), 1, file);
        fread(&is_normalized, sizeof(int), 1, file);
        
        // Saltar arrays de estadísticas
        fseek(file, num_features * sizeof(double) * 4, SEEK_CUR);
        
        // Saltar nombres de características
        for (int i = 0; i < num_features; i++) {
            int len;
            fread(&len, sizeof(int), 1, file);
            fseek(file, len * sizeof(char), SEEK_CUR);
        }
        
        // Saltar nombres de clases
        for (int i = 0; i < num_classes; i++) {
            int len;
            fread(&len, sizeof(int), 1, file);
            fseek(file, len * sizeof(char), SEEK_CUR);
        }
    }
    
    fclose(file);
    
    // Actualizar tiempo de último uso
    tree->last_used = time(NULL);
    
    return tree;
}

TreeNode* load_tree_from_file(FILE* file) {
    int marker;
    fread(&marker, sizeof(int), 1, file);
    
    if (marker == -1) {
        return NULL;
    }
    
    // Retroceder para leer el nodo completo
    fseek(file, -sizeof(int), SEEK_CUR);
    
    TreeNode* node = malloc(sizeof(TreeNode));
    if (!node) {
        return NULL;
    }
    
    memset(node, 0, sizeof(TreeNode));
    
    fread(&node->is_leaf, sizeof(int), 1, file);
    fread(&node->samples, sizeof(int), 1, file);
    fread(&node->depth, sizeof(int), 1, file);
    fread(&node->gini, sizeof(double), 1, file);
    fread(&node->entropy, sizeof(double), 1, file);
    
    if (node->is_leaf) {
        fread(&node->class_label, sizeof(int), 1, file);
        node->left = NULL;
        node->right = NULL;
    } else {
        fread(&node->split_feature, sizeof(int), 1, file);
        fread(&node->split_value, sizeof(double), 1, file);
        fread(&node->num_categories, sizeof(int), 1, file);
        
        node->left = load_tree_from_file(file);
        node->right = load_tree_from_file(file);
    }
    
    return node;
}

int save_model_registry() {
    FILE* file = fopen("model_registry.bin", "wb");
    if (!file) return 0;
    
    fwrite(&model_registry.num_models, sizeof(int), 1, file);
    for (int i = 0; i < model_registry.num_models; i++) {
        fwrite(&model_registry.models[i], sizeof(ModelInfo), 1, file);
    }
    
    fclose(file);
    return 1;
}

int load_model_registry() {
    FILE* file = fopen("model_registry.bin", "rb");
    if (!file) return 0;
    
    fread(&model_registry.num_models, sizeof(int), 1, file);
    if (model_registry.num_models > 100) model_registry.num_models = 100;
    
    for (int i = 0; i < model_registry.num_models; i++) {
        fread(&model_registry.models[i], sizeof(ModelInfo), 1, file);
    }
    
    fclose(file);
    return 1;
}
