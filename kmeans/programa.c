/*******************************************************************************
 * SISTEMA K-MEANS DIDACTICO COMPLETO - CLUSTERING NO SUPERVISADO
 * Sistema educativo completo para aprender K-Means desde cero
 * Caracteristicas:
 * - K-Means con inicialización aleatoria, k-means++ y manual
 * - Visualización en tiempo real de la formación de clusters
 * - Explicaciones detalladas de cada concepto
 * - Modo "aprendizaje activo" con preguntas y respuestas
 * - Simulación de diferentes tipos de datos y clusters
 * - Análisis de calidad de clustering (inercia, silueta)
 * - Exportación de informes completos
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
#include <stdarg.h>
#include <errno.h>
#include <limits.h>

// ============================ CONFIGURACION ============================
#define MAX_POINTS 1000
#define MAX_CLUSTERS 20
#define MAX_FEATURES 10
#define MAX_ITERATIONS 500
#define TERMINAL_WIDTH 80
#define TERMINAL_HEIGHT 40
#define COLOR_RESET "\033[0m"
#define COLOR_RED "\033[31m"
#define COLOR_GREEN "\033[32m"
#define COLOR_YELLOW "\033[33m"
#define COLOR_BLUE "\033[34m"
#define COLOR_MAGENTA "\033[35m"
#define COLOR_CYAN "\033[36m"
#define COLOR_WHITE "\033[37m"
#define COLOR_BRIGHT_BLACK "\033[90m"

// Símbolos para visualización (ASCII para evitar problemas de portabilidad)
#define SYMBOLS "*#@+ox^v<>"
#define BLOCK_CHARS "█▓▒░"

// ============================ ESTRUCTURAS DE DATOS ============================

typedef struct {
    double features[MAX_FEATURES];
    int cluster_id;          // ID del cluster asignado
    double distance_to_centroid;  // Distancia a su centroide
    int is_noise;           // Para DBSCAN (extensión futura)
    int is_boundary;        // Si está en el borde del cluster
} DataPoint;

typedef struct {
    double centroid[MAX_FEATURES];  // Centroide actual
    double prev_centroid[MAX_FEATURES];  // Centroide anterior
    DataPoint points[MAX_POINTS];   // Puntos asignados
    int point_count;
    int id;                         // ID del cluster
    char color_code[10];            // Código de color para visualización
    char symbol;                    // Símbolo para visualización
    double inertia;                 // Suma de cuadrados intra-cluster
    double radius;                  // Radio aproximado del cluster
    int is_stable;                  // Si el centroide se estabilizó
} Cluster;

typedef struct {
    DataPoint points[MAX_POINTS];
    int num_points;
    int num_features;
    char feature_names[MAX_FEATURES][50];
    double feature_min[MAX_FEATURES];
    double feature_max[MAX_FEATURES];
    int is_normalized;
    char name[100];
    char description[256];
} Dataset;

typedef struct {
    Cluster clusters[MAX_CLUSTERS];
    int num_clusters;
    int iterations;
    double total_inertia;
    double silhouette_score;
    double davies_bouldin_score;
    time_t trained_at;
    char name[100];
    char initialization_method[20];
    int num_features_trained;
    double training_time;
    double convergence_threshold;
    int converged;
    double inertia_history[MAX_ITERATIONS];
    double centroids_history[MAX_ITERATIONS][MAX_CLUSTERS][MAX_FEATURES];
} KMeans_Model;

typedef struct {
    double inertia_history[MAX_ITERATIONS];
    double silhouette_history[MAX_ITERATIONS];
    int point_movements_history[MAX_ITERATIONS];
    double centroids_movement[MAX_ITERATIONS];
    int iteration_count;
} TrainingHistory;

typedef struct {
    double silhouette_score;
    double davies_bouldin;
    double calinski_harabasz;
    double inertia;
    double homogeneity;
    double completeness;
    double v_measure;
    int cluster_sizes[MAX_CLUSTERS];
    double cluster_density[MAX_CLUSTERS];
} ClusterMetrics;

typedef struct {
    char question[256];
    char options[4][100];
    int correct_answer;
    char explanation[512];
} QuizQuestion;

// ============================ VARIABLES GLOBALES ============================
Dataset current_dataset = {0};
KMeans_Model current_model = {0};
TrainingHistory training_history = {0};
ClusterMetrics current_metrics = {0};
int terminal_width = TERMINAL_WIDTH;
char current_model_file[256] = "";
char current_dataset_file[256] = "";
int learning_mode = 0;  // 0=normal, 1=paso a paso, 2=examen
int quiz_score = 0;
int total_questions = 0;
QuizQuestion quiz_questions[20];
int animation_speed = 100000;  // Microsegundos entre frames

// ============================ PROTOTIPOS DE FUNCIONES ============================

// Sistema e inicialización
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
void set_animation_speed(int speed);

// Visualización mejorada
void print_color(const char* color, const char* format, ...);
void print_bullet(const char* text, int indent);
void print_section(const char* title);
void print_note(const char* note);
void print_warning(const char* warning);
void print_success(const char* format, ...);
void print_error(const char* format, ...);
void center_text(const char* text);
void draw_box(const char* title, const char* content);
void animate_progress(const char* message, int steps);
void print_clustering_visualization_2d(Dataset* dataset, KMeans_Model* model, int iteration);
void print_clusters_info(KMeans_Model* model);
void print_centroids_movement_animation(KMeans_Model* model);
void print_elbow_method_visualization(KMeans_Model models[], int num_models);
void print_silhouette_visualization(Dataset* dataset, KMeans_Model* model);
void print_cluster_boundaries(Dataset* dataset, KMeans_Model* model);
void print_feature_space_analysis(Dataset* dataset, KMeans_Model* model);
void print_model_info(KMeans_Model* model);
void print_dataset_visualization(Dataset* dataset);

// Manejo de datasets
Dataset load_dataset(const char* filename);
Dataset create_random_clusters_dataset(int points, int clusters, double spread);
Dataset create_circular_clusters_dataset(int points, int circles);
Dataset create_moon_shaped_dataset(int points, int moons);
Dataset create_spiral_dataset(int points, int arms);
Dataset create_blob_dataset(int points, int blobs, double cluster_std);
Dataset create_anisotropic_dataset(int points);
Dataset create_varied_variance_dataset(int points);
Dataset create_no_structure_dataset(int points);
void normalize_dataset(Dataset* dataset);
void print_dataset_info(Dataset* dataset);
void save_dataset(Dataset* dataset, const char* filename);
void add_noise_to_dataset(Dataset* dataset, double noise_level);
void shuffle_dataset(Dataset* dataset);

// Funciones K-Means
double euclidean_distance(double a[], double b[], int n);
int assign_points_to_clusters(Dataset* dataset, KMeans_Model* model);
int update_centroids(Dataset* dataset, KMeans_Model* model);
void initialize_centroids_random(Dataset* dataset, KMeans_Model* model);
void initialize_centroids_kmeansplusplus(Dataset* dataset, KMeans_Model* model);
void initialize_centroids_manual(Dataset* dataset, KMeans_Model* model);
int kmeans_has_converged(KMeans_Model* model, double threshold);
double calculate_inertia(Dataset* dataset, KMeans_Model* model);
double calculate_silhouette_score(Dataset* dataset, KMeans_Model* model);
double calculate_davies_bouldin_score(Dataset* dataset, KMeans_Model* model);

// Entrenamiento
void train_kmeans(Dataset* dataset, KMeans_Model* model, int max_iterations);
void train_kmeans_step_by_step(Dataset* dataset, KMeans_Model* model);
void train_kmeans_with_animation(Dataset* dataset, KMeans_Model* model);
void find_optimal_k_elbow_method(Dataset* dataset, int k_min, int k_max);
void find_optimal_k_silhouette(Dataset* dataset, int k_min, int k_max);

// Evaluación
ClusterMetrics evaluate_clustering(KMeans_Model* model, Dataset* dataset);
void print_cluster_metrics(ClusterMetrics* metrics, KMeans_Model* model);
void compare_clustering_algorithms(Dataset* dataset);

// Persistencia
int save_model(KMeans_Model* model, const char* filename);
int load_model(KMeans_Model* model, const char* filename);
void save_model_interactive(KMeans_Model* model);
void load_model_interactive(KMeans_Model* model);
void export_full_report(KMeans_Model* model, Dataset* dataset, const char* filename);

// Sistema de aprendizaje
void learning_mode_menu();
void interactive_tutorial();
void step_by_step_clustering();
void concept_explanation(const char* concept);
void take_quiz();
void show_quiz_results();
void ask_question(QuizQuestion* question);
void load_quiz_questions();
void explain_clustering_concepts(KMeans_Model* model, Dataset* dataset);

// Interfaz
void interactive_mode();
void training_mode();
void visualization_mode();
void demo_mode();
void tutorial_mode();
void analysis_mode();
void model_management_menu();
void dataset_management_menu();
void settings_mode();

// Utilidades
double random_double(double min, double max);
int random_int(int min, int max);
void sleep_ms(int milliseconds);

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
        else if (strcmp(argv[i], "-fast") == 0) animation_speed = 50000;
        else if (strcmp(argv[i], "-slow") == 0) animation_speed = 200000;
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
    
    // Cargar modelo si se especificó
    if (model_file) {
        print_color(COLOR_CYAN, "Cargando modelo: %s\n", model_file);
        if (load_model(&current_model, model_file)) {
            strcpy(current_model_file, model_file);
            print_success("Modelo K-Means cargado exitosamente!");
            print_model_info(&current_model);
            
            // Si hay dataset, evaluar el modelo
            if (data_file) {
                print_color(COLOR_CYAN, "\nCargando dataset: %s\n", data_file);
                current_dataset = load_dataset(data_file);
                if (current_dataset.num_points > 0) {
                    strcpy(current_dataset_file, data_file);
                    normalize_dataset(&current_dataset);
                }
            }
        } else {
            print_error("No se pudo cargar el modelo. Inicializando nuevo modelo.");
        }
    }
    
    // Cargar dataset si se especificó
    if (data_file && !model_file) {
        print_color(COLOR_CYAN, "Cargando dataset: %s\n", data_file);
        current_dataset = load_dataset(data_file);
        if (current_dataset.num_points == 0) {
            print_warning("Error al cargar. Generando dataset de ejemplo.");
            current_dataset = create_random_clusters_dataset(150, 3, 1.5);
            strcpy(current_dataset.name, "Dataset de Clusters Aleatorios");
        } else {
            strcpy(current_dataset_file, data_file);
        }
    } else if (!model_file) {
        print_color(COLOR_CYAN, "Generando dataset de ejemplo...\n");
        current_dataset = create_random_clusters_dataset(150, 3, 1.5);
        strcpy(current_dataset.name, "Dataset de 3 Clusters");
    }
    
    // Normalizar dataset si existe
    if (current_dataset.num_points > 0) {
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
    print_color(COLOR_CYAN, "🚀 Inicializando Sistema K-Means Didáctico...\n");
    
    srand(time(NULL));
    terminal_width = TERMINAL_WIDTH;
    
    // Inicializar variables globales
    memset(current_model_file, 0, sizeof(current_model_file));
    memset(current_dataset_file, 0, sizeof(current_dataset_file));
    learning_mode = 0;
    quiz_score = 0;
    total_questions = 0;
    animation_speed = 100000;  // 100ms por defecto
    
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
    print_color(COLOR_YELLOW, "🧹 Finalizando Sistema K-Means Didáctico\n");
    
    // Mostrar resumen si hay modelo entrenado
    if (current_model.num_clusters > 0) {
        printf("\nResumen de la sesión:\n");
        printf("  • Modelo: %s\n", current_model.name);
        printf("  • Clusters: %d\n", current_model.num_clusters);
        printf("  • Inercia: %.4f\n", current_model.total_inertia);
        printf("  • Puntuación de silueta: %.4f\n", current_model.silhouette_score);
        
        if (learning_mode) {
            printf("  • Puntaje del quiz: %d/%d\n", quiz_score, total_questions);
        }
    }
    
    printf("\n¡Gracias por usar el Sistema K-Means Didáctico!\n");
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

void print_success(const char* format, ...) {
    va_list args;
    printf("%s✅ ", COLOR_GREEN);
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    printf("%s\n", COLOR_RESET);
}

void print_error(const char* format, ...) {
    va_list args;
    printf("%s", COLOR_RED);
    printf("❌ Error: ");
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    printf("%s\n", COLOR_RESET);
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
    center_text("🤖 SISTEMA K-MEANS DIDÁCTICO - APRENDE CLUSTERING");
    print_separator('=');
    
    printf("\n");
    center_text("Una herramienta educativa completa para entender K-Means desde cero");
    printf("\n");
    
    print_color(COLOR_MAGENTA, "🎯 Características principales:\n");
    print_bullet("Clustering visual paso a paso", 1);
    print_bullet("Explicaciones detalladas de cada concepto", 1);
    print_bullet("Modo aprendizaje con preguntas y respuestas", 1);
    print_bullet("Análisis de calidad de clustering", 1);
    print_bullet("Persistencia de modelos y datasets", 1);
    print_bullet("Generación de informes completos", 1);
    print_bullet("Múltiples métodos de inicialización", 1);
    print_bullet("Visualización de límites de clusters", 1);
    
    printf("\n");
    print_color(COLOR_YELLOW, "💡 Consejo: Usa el modo -learn para una experiencia educativa guiada.\n");
    printf("            Usa -fast para animaciones rápidas o -slow para lentas.\n");
    printf("\n");
}

void setup_learning_environment() {
    print_header("🎓 MODO APRENDIZAJE ACTIVO");
    
    printf("\nBienvenido al modo aprendizaje activo. En este modo:\n");
    print_bullet("Cada concepto se explica detalladamente", 1);
    print_bullet("Podrás ver el clustering paso a paso", 1);
    print_bullet("Responderás preguntas para reforzar tu comprensión", 1);
    print_bullet("Analizarás errores comunes", 1);
    print_bullet("Obtendrás recomendaciones personalizadas", 1);
    
    printf("\n");
    print_color(COLOR_GREEN, "¿Estás listo para comenzar tu aprendizaje? (s/n): ");
    
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

void set_animation_speed(int speed) {
    animation_speed = speed;
    printf("Velocidad de animación ajustada a %d ms por frame\n", speed / 1000);
}

void animate_progress(const char* message, int steps) {
    printf("%s [", message);
    for (int i = 0; i < steps; i++) {
        printf(".");
        fflush(stdout);
        usleep(animation_speed);
    }
    printf("] Listo!\n");
}

double random_double(double min, double max) {
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

int random_int(int min, int max) {
    return min + rand() % (max - min + 1);
}

void sleep_ms(int milliseconds) {
    usleep(milliseconds * 1000);
}

// ============================ VISUALIZACIÓN ============================

void print_clustering_visualization_2d(Dataset* dataset, KMeans_Model* model, int iteration) {
    if (dataset->num_features < 2) {
        print_error("Se necesitan al menos 2 características para visualizar");
        return;
    }
    
    // Configurar colores y símbolos para clusters
    const char* cluster_colors[] = {COLOR_RED, COLOR_GREEN, COLOR_BLUE, COLOR_YELLOW, 
                                   COLOR_MAGENTA, COLOR_CYAN, COLOR_WHITE};
    const char cluster_symbols[] = {'*', '#', '@', '+', 'x', 'o', 's', 'd', 'v', '^'};
    
    clear_screen();
    print_header("VISUALIZACIÓN K-MEANS - ITERACIÓN");
    printf("Iteración: %d | Clusters: %d | Inercia: %.4f\n\n", 
           iteration, model->num_clusters, model->total_inertia);
    
    int grid_size = 60;
    int grid_height = 30;
    char grid[grid_height][grid_size];
    
    // Inicializar grid con espacios
    for (int i = 0; i < grid_height; i++) {
        for (int j = 0; j < grid_size; j++) {
            grid[i][j] = ' ';
        }
    }
    
    // Calcular límites del dataset
    double min_x = dataset->feature_min[0];
    double max_x = dataset->feature_max[0];
    double min_y = dataset->feature_min[1];
    double max_y = dataset->feature_max[1];
    
    // Dibujar ejes
    for (int i = 0; i < grid_height; i++) {
        int x_pos = (int)((0 - min_x) / (max_x - min_x) * (grid_size - 1));
        if (x_pos >= 0 && x_pos < grid_size) {
            grid[i][x_pos] = '|';
        }
    }
    
    for (int j = 0; j < grid_size; j++) {
        int y_pos = (int)((0 - min_y) / (max_y - min_y) * (grid_height - 1));
        y_pos = grid_height - 1 - y_pos;
        if (y_pos >= 0 && y_pos < grid_height) {
            grid[y_pos][j] = '-';
        }
    }
    
    // Dibujar puntos del dataset
    for (int p = 0; p < dataset->num_points && p < 200; p++) {
        int x = (int)((dataset->points[p].features[0] - min_x) / (max_x - min_x) * (grid_size - 1));
        int y = (int)((dataset->points[p].features[1] - min_y) / (max_y - min_y) * (grid_height - 1));
        y = grid_height - 1 - y;
        
        if (x >= 0 && x < grid_size && y >= 0 && y < grid_height) {
            int cluster_id = dataset->points[p].cluster_id;
            if (cluster_id >= 0 && cluster_id < model->num_clusters) {
                grid[y][x] = cluster_symbols[cluster_id % 10];
            } else {
                grid[y][x] = '.';
            }
        }
    }
    
    // Dibujar centroides
    for (int c = 0; c < model->num_clusters; c++) {
        int x = (int)((model->clusters[c].centroid[0] - min_x) / (max_x - min_x) * (grid_size - 1));
        int y = (int)((model->clusters[c].centroid[1] - min_y) / (max_y - min_y) * (grid_height - 1));
        y = grid_height - 1 - y;
        
        if (x >= 0 && x < grid_size && y >= 0 && y < grid_height) {
            grid[y][x] = 'X';  // Centroide
        }
    }
    
    // Imprimir grid con colores
    printf("    y\n");
    printf("    ↑\n");
    for (int i = 0; i < grid_height; i++) {
        printf("%3.1f│", max_y - (max_y - min_y) * i / (grid_height - 1));
        
        for (int j = 0; j < grid_size; j++) {
            char c = grid[i][j];
            
            // Determinar color basado en el carácter
            if (c == 'X') {
                printf("\033[1;37mX\033[0m");  // Centroides en blanco brillante
            } else if (c >= '*' && c <= '^') {
                // Encontrar a qué cluster pertenece el símbolo
                for (int k = 0; k < model->num_clusters; k++) {
                    if (cluster_symbols[k % 10] == c) {
                        printf("%s%c%s", cluster_colors[k % 7], c, COLOR_RESET);
                        break;
                    }
                }
            } else if (c == '|' || c == '-') {
                printf("\033[90m%c\033[0m", c);  // Ejes en gris
            } else {
                printf("%c", c);
            }
        }
        printf("\n");
    }
    
    // Eje X
    printf("    └");
    for (int j = 0; j < grid_size; j++) printf("─");
    printf("→ x\n     ");
    
    for (int j = 0; j < grid_size; j += 10) {
        printf("%-8.1f", min_x + (max_x - min_x) * j / (grid_size - 1));
    }
    printf("\n");
    
    // Leyenda
    printf("\n📊 Leyenda:\n");
    for (int c = 0; c < model->num_clusters; c++) {
        printf("  %s%c%s = Cluster %d (%d puntos)", 
               cluster_colors[c % 7], 
               cluster_symbols[c % 10],
               COLOR_RESET,
               c + 1,
               model->clusters[c].point_count);
        if (c < model->num_clusters - 1) printf(" | ");
        if ((c + 1) % 3 == 0) printf("\n");
    }
    printf("\n  \033[1;37mX\033[0m = Centroides\n");
    
    // Información de la iteración
    printf("\n📈 Información de la iteración:\n");
    printf("  • Puntos reasignados: %d\n", training_history.point_movements_history[iteration]);
    printf("  • Movimiento centroides: %.4f\n", training_history.centroids_movement[iteration]);
    printf("  • Inercia: %.4f\n", model->total_inertia);
    
    if (iteration > 0) {
        double improvement = training_history.inertia_history[iteration-1] - model->total_inertia;
        printf("  • Mejora inercia: %.4f (%.2f%%)\n", 
               improvement, 
               improvement/training_history.inertia_history[iteration-1]*100);
    }
}

void print_clusters_info(KMeans_Model* model) {
    print_section("INFORMACIÓN DETALLADA DE CLUSTERS");
    
    printf("┌─────┬──────────┬────────────┬────────────┬────────────┬────────────┐\n");
    printf("│  #  │  Puntos  │   Radio    │   Inercia  │  Estable   │  Densidad  │\n");
    printf("├─────┼──────────┼────────────┼────────────┼────────────┼────────────┤\n");
    
    double total_points = 0;
    for (int i = 0; i < model->num_clusters; i++) {
        total_points += model->clusters[i].point_count;
    }
    
    for (int i = 0; i < model->num_clusters; i++) {
        double percentage = (total_points > 0) ? 
            (double)model->clusters[i].point_count / total_points * 100 : 0;
        double density = (model->clusters[i].radius > 0) ? 
            model->clusters[i].point_count / model->clusters[i].radius : 0;
        
        printf("│ %3d │ %8d │ %10.4f │ %10.4f │ %10s │ %10.2f │\n",
               i + 1,
               model->clusters[i].point_count,
               model->clusters[i].radius,
               model->clusters[i].inertia,
               model->clusters[i].is_stable ? "Sí" : "No",
               density);
    }
    
    printf("└─────┴──────────┴────────────┴────────────┴────────────┴────────────┘\n");
    
    // Estadísticas generales
    printf("\n📊 Estadísticas generales:\n");
    printf("  • Total puntos: %.0f\n", total_points);
    printf("  • Inercia total: %.4f\n", model->total_inertia);
    printf("  • Inercia promedio por cluster: %.4f\n", 
           model->total_inertia / model->num_clusters);
    
    // Distribución de puntos
    printf("\n📈 Distribución de puntos por cluster:\n");
    for (int i = 0; i < model->num_clusters; i++) {
        double percentage = (total_points > 0) ? 
            (double)model->clusters[i].point_count / total_points * 100 : 0;
        
        printf("  Cluster %d: ", i + 1);
        int bar_length = (int)(percentage / 2);
        for (int j = 0; j < bar_length; j++) printf("█");
        for (int j = bar_length; j < 50; j++) printf(" ");
        printf(" %5.1f%% (%d puntos)\n", percentage, model->clusters[i].point_count);
    }
    
    // Análisis de calidad
    printf("\n🔍 Análisis de calidad:\n");
    if (model->silhouette_score > 0.7) {
        printf("  • ✅ Excelente estructura de clusters (silueta > 0.7)\n");
    } else if (model->silhouette_score > 0.5) {
        printf("  • ⚠️  Estructura razonable (silueta > 0.5)\n");
    } else if (model->silhouette_score > 0.25) {
        printf("  • ⚠️  Estructura débil (silueta > 0.25)\n");
    } else {
        printf("  • ❌ Sin estructura clara (silueta ≤ 0.25)\n");
    }
    
    // Detección de clusters problemáticos
    printf("\n🎯 Clusters potencialmente problemáticos:\n");
    int problem_clusters = 0;
    for (int i = 0; i < model->num_clusters; i++) {
        if (model->clusters[i].point_count < 5) {
            printf("  • Cluster %d: Muy pocos puntos (%d)\n", i + 1, model->clusters[i].point_count);
            problem_clusters++;
        }
        if (model->clusters[i].radius > 2.0) {
            printf("  • Cluster %d: Radio muy grande (%.2f)\n", i + 1, model->clusters[i].radius);
            problem_clusters++;
        }
    }
    
    if (problem_clusters == 0) {
        printf("  • ✅ Todos los clusters parecen saludables\n");
    }
}

void print_centroids_movement_animation(KMeans_Model* model) {
    if (model->iterations < 2) {
        print_warning("No hay suficiente historial de movimiento");
        return;
    }
    
    print_section("ANIMACIÓN DE MOVIMIENTO DE CENTROIDES");
    
    printf("Esta animación muestra cómo se mueven los centroides durante el entrenamiento.\n");
    printf("Cada iteración muestra la posición de los centroides y cómo convergen.\n\n");
    
    wait_for_key("Presiona Enter para comenzar la animación...");
    
    // Configurar grid para animación
    int grid_size = 50;
    int grid_height = 25;
    
    // Encontrar límites del espacio
    double min_x = 1e10, max_x = -1e10, min_y = 1e10, max_y = -1e10;
    
    for (int iter = 0; iter < model->iterations; iter++) {
        for (int c = 0; c < model->num_clusters; c++) {
            double x = model->centroids_history[iter][c][0];
            double y = model->centroids_history[iter][c][1];
            
            if (x < min_x) min_x = x;
            if (x > max_x) max_x = x;
            if (y < min_y) min_y = y;
            if (y > max_y) max_y = y;
        }
    }
    
    // Añadir margen
    double x_range = max_x - min_x;
    double y_range = max_y - min_y;
    min_x -= x_range * 0.1;
    max_x += x_range * 0.1;
    min_y -= y_range * 0.1;
    max_y += y_range * 0.1;
    
    // Animación iteración por iteración
    for (int iter = 0; iter < model->iterations; iter++) {
        clear_screen();
        printf("Iteración %d/%d - Movimiento de Centroides\n\n", iter + 1, model->iterations);
        
        char grid[grid_height][grid_size];
        for (int i = 0; i < grid_height; i++) {
            for (int j = 0; j < grid_size; j++) {
                grid[i][j] = ' ';
            }
        }
        
        // Dibujar trayectorias de centroides
        for (int c = 0; c < model->num_clusters; c++) {
            const char* colors[] = {COLOR_RED, COLOR_GREEN, COLOR_BLUE, COLOR_YELLOW, 
                                   COLOR_MAGENTA, COLOR_CYAN};
            
            // Dibujar camino hasta esta iteración
            for (int t = 0; t < iter; t++) {
                if (t + 1 < model->iterations) {
                    int x1 = (int)((model->centroids_history[t][c][0] - min_x) / 
                                  (max_x - min_x) * (grid_size - 1));
                    int y1 = (int)((model->centroids_history[t][c][1] - min_y) / 
                                  (max_y - min_y) * (grid_height - 1));
                    y1 = grid_height - 1 - y1;
                    
                    int x2 = (int)((model->centroids_history[t+1][c][0] - min_x) / 
                                  (max_x - min_x) * (grid_size - 1));
                    int y2 = (int)((model->centroids_history[t+1][c][1] - min_y) / 
                                  (max_y - min_y) * (grid_height - 1));
                    y2 = grid_height - 1 - y2;
                    
                    // Dibujar línea entre puntos consecutivos
                    if (x1 >= 0 && x1 < grid_size && y1 >= 0 && y1 < grid_height &&
                        x2 >= 0 && x2 < grid_size && y2 >= 0 && y2 < grid_height) {
                        
                        // Algoritmo simple de línea
                        int dx = abs(x2 - x1);
                        int dy = abs(y2 - y1);
                        int sx = (x1 < x2) ? 1 : -1;
                        int sy = (y1 < y2) ? 1 : -1;
                        int err = dx - dy;
                        
                        while (1) {
                            if (x1 >= 0 && x1 < grid_size && y1 >= 0 && y1 < grid_height) {
                                grid[y1][x1] = '.';
                            }
                            
                            if (x1 == x2 && y1 == y2) break;
                            
                            int e2 = 2 * err;
                            if (e2 > -dy) {
                                err -= dy;
                                x1 += sx;
                            }
                            if (e2 < dx) {
                                err += dx;
                                y1 += sy;
                            }
                        }
                    }
                }
            }
            
            // Dibujar centroide actual
            int x = (int)((model->centroids_history[iter][c][0] - min_x) / 
                         (max_x - min_x) * (grid_size - 1));
            int y = (int)((model->centroids_history[iter][c][1] - min_y) / 
                         (max_y - min_y) * (grid_height - 1));
            y = grid_height - 1 - y;
            
            if (x >= 0 && x < grid_size && y >= 0 && y < grid_height) {
                grid[y][x] = '0' + (c + 1) % 10;  // Número del cluster
            }
        }
        
        // Imprimir grid
        for (int i = 0; i < grid_height; i++) {
            printf("   ");
            for (int j = 0; j < grid_size; j++) {
                char c = grid[i][j];
                if (c >= '0' && c <= '9') {
                    int cluster_num = c - '0';
                    const char* colors[] = {COLOR_RED, COLOR_GREEN, COLOR_BLUE, COLOR_YELLOW, 
                                           COLOR_MAGENTA, COLOR_CYAN};
                    printf("%s%c%s", colors[cluster_num % 6], c, COLOR_RESET);
                } else if (c == '.') {
                    printf("\033[90m.\033[0m");
                } else {
                    printf("%c", c);
                }
            }
            printf("\n");
        }
        
        // Información
        printf("\nLeyenda: Números = Centroides, Puntos = Trayectoria\n");
        printf("Inercia actual: %.4f\n", model->inertia_history[iter]);
        
        if (iter < model->iterations - 1) {
            printf("\nSiguiente iteración en 0.5 segundos...\n");
            usleep(500000);
        }
    }
    
    printf("\n¡Animación completada!\n");
    wait_for_enter();
}

void print_elbow_method_visualization(KMeans_Model models[], int num_models) {
    print_section("MÉTODO DEL CODO - SELECCIÓN DE K ÓPTIMO");
    
    printf("El método del codo ayuda a seleccionar el número óptimo de clusters (K).\n");
    printf("Se busca el punto donde agregar más clusters ya no reduce significativamente la inercia.\n\n");
    
    // Encontrar valores máximos y mínimos para escalar
    double max_inertia = 0;
    double min_inertia = 1e10;
    
    for (int i = 0; i < num_models; i++) {
        if (models[i].total_inertia > max_inertia) max_inertia = models[i].total_inertia;
        if (models[i].total_inertia < min_inertia) min_inertia = models[i].total_inertia;
    }
    
    // Crear gráfico
    int graph_height = 20;
    int graph_width = 60;
    
    printf("Gráfico de Inercia vs Número de Clusters (K):\n\n");
    printf("Inercia\n");
    printf("  ↑\n");
    
    for (int h = graph_height; h >= 0; h--) {
        printf("%6.1f │", max_inertia - (max_inertia - min_inertia) * h / graph_height);
        
        for (int k = 1; k <= num_models; k++) {
            int x_pos = (int)((k - 1.0) / (num_models - 1) * (graph_width - 1));
            
            // Calcular posición de este valor de K
            double normalized_inertia = (models[k-1].total_inertia - min_inertia) / 
                                       (max_inertia - min_inertia);
            int y_pos = (int)(normalized_inertia * graph_height);
            
            if (h == y_pos) {
                printf("●");  // Punto de datos
            } else if (h == 0) {
                printf("─");  // Eje X
            } else {
                printf(" ");
            }
        }
        printf("\n");
    }
    
    // Eje X
    printf("       └");
    for (int i = 0; i < graph_width; i++) printf("─");
    printf("→ K\n        ");
    
    for (int k = 1; k <= num_models; k++) {
        if (k % 2 == 1 || k == num_models || k == 1) {
            printf("%-3d", k);
        } else {
            printf("   ");
        }
    }
    printf("\n");
    
    // Mostrar valores
    printf("\n📊 Valores de inercia:\n");
    printf("┌─────┬────────────┬──────────────┐\n");
    printf("│  K  │   Inercia  │  Reducción  │\n");
    printf("├─────┼────────────┼──────────────┤\n");
    
    double prev_inertia = models[0].total_inertia;
    for (int i = 0; i < num_models; i++) {
        double reduction = (i == 0) ? 0 : 
            (prev_inertia - models[i].total_inertia) / prev_inertia * 100;
        
        printf("│ %3d │ %10.2f │ %10.1f%% │\n", 
               i + 1, 
               models[i].total_inertia,
               reduction);
        
        prev_inertia = models[i].total_inertia;
    }
    printf("└─────┴────────────┴──────────────┘\n");
    
    // Encontrar el codo (punto de mayor curvatura)
    double max_curvature = -1e10;
    int elbow_k = 2;
    
    for (int k = 2; k < num_models; k++) {
        double y1 = models[k-2].total_inertia;
        double y2 = models[k-1].total_inertia;
        double y3 = models[k].total_inertia;
        
        // Calcular curvatura aproximada
        double curvature = fabs((y3 - 2*y2 + y1) / (y1 - y3));
        
        if (curvature > max_curvature) {
            max_curvature = curvature;
            elbow_k = k;
        }
    }
    
    printf("\n🎯 Análisis del método del codo:\n");
    printf("  • K sugerido por el método del codo: %d\n", elbow_k + 1);
    printf("  • Inercia en K=%d: %.2f\n", elbow_k + 1, models[elbow_k].total_inertia);
    printf("  • Reducción respecto a K=1: %.1f%%\n", 
           (models[0].total_inertia - models[elbow_k].total_inertia) / 
           models[0].total_inertia * 100);
    
    // Mostrar silueta también si está disponible
    printf("\n📈 Puntuaciones de silueta para referencia:\n");
    for (int i = 0; i < num_models && i < 8; i++) {
        printf("  K=%d: Silueta=%.3f", i + 1, models[i].silhouette_score);
        if (i + 1 == elbow_k + 1) printf(" ← Codo sugerido");
        printf("\n");
    }
    
    // Recomendación
    printf("\n💡 Recomendación:\n");
    printf("  Basado en el método del codo, considera usar K = %d\n", elbow_k + 1);
    printf("  Sin embargo, también considera:\n");
    printf("  1. El conocimiento del dominio\n");
    printf("  2. La puntuación de silueta\n");
    printf("  3. La interpretabilidad de los clusters\n");
}

void print_silhouette_visualization(Dataset* dataset, KMeans_Model* model) {
    print_section("ANÁLISIS DE SILUETA POR PUNTO");
    
    printf("La silueta mide qué tan similar es un punto a su propio cluster\n");
    printf("comparado con otros clusters. Valores cercanos a 1 indican buena asignación.\n\n");
    
    // Calcular silueta para cada punto
    double silhouette_values[MAX_POINTS] = {0};
    int cluster_counts[MAX_CLUSTERS] = {0};
    
    for (int i = 0; i < dataset->num_points; i++) {
        int cluster_i = dataset->points[i].cluster_id;
        
        if (cluster_counts[cluster_i] == 0) {
            // Calcular distancia promedio a puntos en el mismo cluster
            double a_i = 0.0;
            int count_a = 0;
            
            for (int j = 0; j < dataset->num_points; j++) {
                if (i != j && dataset->points[j].cluster_id == cluster_i) {
                    a_i += euclidean_distance(dataset->points[i].features,
                                            dataset->points[j].features,
                                            dataset->num_features);
                    count_a++;
                }
            }
            a_i = (count_a > 0) ? a_i / count_a : 0;
            
            // Calcular distancia mínima promedio a otros clusters
            double b_i = 1e10;
            
            for (int c = 0; c < model->num_clusters; c++) {
                if (c != cluster_i) {
                    double avg_dist = 0.0;
                    int count_b = 0;
                    
                    for (int j = 0; j < dataset->num_points; j++) {
                        if (dataset->points[j].cluster_id == c) {
                            avg_dist += euclidean_distance(dataset->points[i].features,
                                                         dataset->points[j].features,
                                                         dataset->num_features);
                            count_b++;
                        }
                    }
                    avg_dist = (count_b > 0) ? avg_dist / count_b : 0;
                    
                    if (avg_dist < b_i) b_i = avg_dist;
                }
            }
            
            // Calcular silueta
            if (a_i == 0 && b_i == 0) {
                silhouette_values[i] = 0;
            } else {
                silhouette_values[i] = (b_i - a_i) / fmax(a_i, b_i);
            }
        }
        cluster_counts[cluster_i]++;
    }
    
    // Crear histograma de valores de silueta
    printf("Distribución de valores de silueta:\n\n");
    
    int hist[10] = {0};  // 10 bins de -1 a 1
    for (int i = 0; i < dataset->num_points; i++) {
        int bin = (int)((silhouette_values[i] + 1.0) * 5);  // Convertir -1..1 a 0..10
        if (bin < 0) bin = 0;
        if (bin > 9) bin = 9;
        hist[bin]++;
    }
    
    int max_count = 0;
    for (int i = 0; i < 10; i++) {
        if (hist[i] > max_count) max_count = hist[i];
    }
    
    // Imprimir histograma
    printf("  Silueta   Cantidad\n");
    printf("  ────────  ─────────────────────────────────────────────\n");
    
    for (int i = 0; i < 10; i++) {
        double sil_start = -1.0 + i * 0.2;
        double sil_end = -0.8 + i * 0.2;
        
        printf("  %5.1f-%5.1f  ", sil_start, sil_end);
        
        int bar_length = (max_count > 0) ? (hist[i] * 40 / max_count) : 0;
        for (int j = 0; j < bar_length; j++) {
            if (sil_start >= 0.5) printf("█");
            else if (sil_start >= 0.25) printf("▓");
            else if (sil_start >= 0) printf("▒");
            else printf("░");
        }
        printf(" %d\n", hist[i]);
    }
    
    // Interpretación
    printf("\n🎯 Interpretación de la silueta:\n");
    printf("  • > 0.7: Estructura fuerte de clusters\n");
    printf("  • 0.5-0.7: Estructura razonable\n");
    printf("  • 0.25-0.5: Estructura débil\n");
    printf("  • < 0.25: Sin estructura significativa\n");
    printf("  • Negativos: Posible asignación incorrecta\n");
    
    // Calcular promedio por cluster
    printf("\n📊 Silueta promedio por cluster:\n");
    double cluster_silhouette[MAX_CLUSTERS] = {0};
    int cluster_counts_sil[MAX_CLUSTERS] = {0};
    
    for (int i = 0; i < dataset->num_points; i++) {
        int cluster_id = dataset->points[i].cluster_id;
        cluster_silhouette[cluster_id] += silhouette_values[i];
        cluster_counts_sil[cluster_id]++;
    }
    
    for (int c = 0; c < model->num_clusters; c++) {
        if (cluster_counts_sil[c] > 0) {
            double avg_sil = cluster_silhouette[c] / cluster_counts_sil[c];
            printf("  Cluster %d: %.3f", c + 1, avg_sil);
            
            if (avg_sil > 0.7) printf(" ✅ Excelente\n");
            else if (avg_sil > 0.5) printf(" ⚠️  Aceptable\n");
            else if (avg_sil > 0.25) printf(" ⚠️  Débil\n");
            else printf(" ❌ Problemático\n");
        }
    }
    
    // Puntos problemáticos (silueta negativa)
    int negative_silhouette = 0;
    for (int i = 0; i < dataset->num_points; i++) {
        if (silhouette_values[i] < 0) negative_silhouette++;
    }
    
    printf("\n🔍 Puntos potencialmente mal asignados: %d (%.1f%%)\n",
           negative_silhouette,
           (double)negative_silhouette / dataset->num_points * 100);
    
    if (negative_silhouette > dataset->num_points * 0.1) {
        printf("  ⚠️  Muchos puntos con silueta negativa\n");
        printf("  Considera: Cambiar K o método de inicialización\n");
    }
}

void print_cluster_boundaries(Dataset* dataset, KMeans_Model* model) {
    if (dataset->num_features != 2) {
        print_warning("La visualización de límites solo está disponible para 2D");
        return;
    }
    
    print_section("LÍMITES DE DECISIÓN ENTRE CLUSTERS");
    
    printf("Esta visualización muestra las regiones de decisión entre clusters.\n");
    printf("Cada región representa el área donde un punto sería asignado a ese cluster.\n\n");
    
    int grid_size = 60;
    int grid_height = 30;
    
    // Calcular límites
    double min_x = dataset->feature_min[0];
    double max_x = dataset->feature_max[0];
    double min_y = dataset->feature_min[1];
    double max_y = dataset->feature_max[1];
    
    // Expandir un poco los límites
    double x_range = max_x - min_x;
    double y_range = max_y - min_y;
    min_x -= x_range * 0.1;
    max_x += x_range * 0.1;
    min_y -= y_range * 0.1;
    max_y += y_range * 0.1;
    
    printf("Calculando límites de decisión...\n");
    
    // Caracteres para diferentes regiones (más suaves para bordes)
    const char* region_chars = " .:oO";
    const char* cluster_chars = "123456789ABCDEFGHIJ";
    
    // Crear grid
    char grid[grid_height][grid_size];
    double confidence[grid_height][grid_size];
    
    for (int i = 0; i < grid_height; i++) {
        for (int j = 0; j < grid_size; j++) {
            grid[i][j] = ' ';
            confidence[i][j] = 0;
        }
    }
    
    // Para cada celda del grid, determinar el cluster más cercano
    for (int i = 0; i < grid_height; i++) {
        for (int j = 0; j < grid_size; j++) {
            double x = min_x + (max_x - min_x) * j / (grid_size - 1);
            double y = min_y + (max_y - min_y) * (grid_height - 1 - i) / (grid_height - 1);
            
            double test_point[MAX_FEATURES] = {x, y};
            
            // Encontrar el centroide más cercano
            double min_dist = 1e10;
            int closest_cluster = -1;
            double second_min_dist = 1e10;
            
            for (int c = 0; c < model->num_clusters; c++) {
                double dist = euclidean_distance(test_point, 
                                                model->clusters[c].centroid, 
                                                2);
                
                if (dist < min_dist) {
                    second_min_dist = min_dist;
                    min_dist = dist;
                    closest_cluster = c;
                } else if (dist < second_min_dist) {
                    second_min_dist = dist;
                }
            }
            
            // Calcular "confianza" (diferencia entre las dos distancias más cercanas)
            confidence[i][j] = (second_min_dist - min_dist) / (second_min_dist + min_dist + 1e-10);
            
            // Asignar carácter basado en la confianza
            int char_index = (int)(confidence[i][j] * 4);  // 0-4
            if (char_index < 0) char_index = 0;
            if (char_index > 4) char_index = 4;
            
            if (closest_cluster < model->num_clusters) {
                grid[i][j] = region_chars[char_index];
            }
        }
    }
    
    // Ahora dibujar los puntos reales
    for (int p = 0; p < dataset->num_points && p < 100; p++) {
        int x = (int)((dataset->points[p].features[0] - min_x) / (max_x - min_x) * (grid_size - 1));
        int y = (int)((dataset->points[p].features[1] - min_y) / (max_y - min_y) * (grid_height - 1));
        y = grid_height - 1 - y;
        
        if (x >= 0 && x < grid_size && y >= 0 && y < grid_height) {
            int cluster_id = dataset->points[p].cluster_id;
            if (cluster_id >= 0 && cluster_id < model->num_clusters) {
                grid[y][x] = cluster_chars[cluster_id % strlen(cluster_chars)];
            }
        }
    }
    
    // Dibujar centroides
    for (int c = 0; c < model->num_clusters; c++) {
        int x = (int)((model->clusters[c].centroid[0] - min_x) / (max_x - min_x) * (grid_size - 1));
        int y = (int)((model->clusters[c].centroid[1] - min_y) / (max_y - min_y) * (grid_height - 1));
        y = grid_height - 1 - y;
        
        if (x >= 0 && x < grid_size && y >= 0 && y < grid_height) {
            grid[y][x] = 'X';
        }
    }
    
    // Imprimir grid con colores
    const char* cluster_colors[] = {COLOR_RED, COLOR_GREEN, COLOR_BLUE, COLOR_YELLOW, 
                                   COLOR_MAGENTA, COLOR_CYAN, COLOR_WHITE};
    
    printf("\nMapa de regiones de decisión:\n");
    printf("    y\n");
    printf("    ↑\n");
    
    for (int i = 0; i < grid_height; i++) {
        printf("%4.1f│", max_y - (max_y - min_y) * i / (grid_height - 1));
        
        for (int j = 0; j < grid_size; j++) {
            char c = grid[i][j];
            
            if (c == 'X') {
                printf("\033[1;37mX\033[0m");  // Centroides en blanco brillante
            } else if ((c >= '1' && c <= '9') || (c >= 'A' && c <= 'J')) {
                // Es un punto de datos
                int cluster_id = 0;
                if (c >= '1' && c <= '9') cluster_id = c - '1';
                else if (c >= 'A' && c <= 'J') cluster_id = c - 'A' + 9;
                
                if (cluster_id < model->num_clusters) {
                    printf("%s%c%s", cluster_colors[cluster_id % 7], c, COLOR_RESET);
                } else {
                    printf("%c", c);
                }
            } else if (strchr(region_chars, c) != NULL) {
                // Es una región de decisión
                int char_index = strchr(region_chars, c) - region_chars;
                int intensity = 232 + char_index * 6;  // Escala de grises
                printf("\033[38;5;%dm%c\033[0m", intensity, c);
            } else {
                printf("%c", c);
            }
        }
        printf("\n");
    }
    
    // Eje X
    printf("    └");
    for (int j = 0; j < grid_size; j++) printf("─");
    printf("→ x\n     ");
    
    for (int j = 0; j < grid_size; j += 10) {
        printf("%-8.1f", min_x + (max_x - min_x) * j / (grid_size - 1));
    }
    printf("\n");
    
    // Leyenda
    printf("\n📊 Leyenda:\n");
    printf("  Números/Letras = Puntos de datos (cada cluster diferente)\n");
    printf("  X = Centroides\n");
    printf("  ");
    for (int i = 0; i < 5; i++) {
        int intensity = 232 + i * 6;
        printf("\033[38;5;%dm%c\033[0m", intensity, region_chars[i]);
        if (i < 4) printf(" → ");
    }
    printf(" = Bordes más definidos → Centro del cluster\n");
    
    // Análisis de los límites
    printf("\n🔍 Análisis de los límites de decisión:\n");
    
    // Calcular "nitidez" de los límites
    int boundary_cells = 0;
    int total_cells = grid_size * grid_height;
    
    for (int i = 0; i < grid_height; i++) {
        for (int j = 0; j < grid_size; j++) {
            if (confidence[i][j] < 0.3) boundary_cells++;
        }
    }
    
    double boundary_ratio = (double)boundary_cells / total_cells;
    
    printf("  • Área de bordes difusos: %.1f%%\n", boundary_ratio * 100);
    
    if (boundary_ratio > 0.3) {
        printf("  ⚠️  Límites muy difusos - los clusters se superponen mucho\n");
    } else if (boundary_ratio > 0.1) {
        printf("  ⚠️  Límites moderadamente definidos\n");
    } else {
        printf("  ✅ Límites bien definidos - clusters separados\n");
    }
}

void print_feature_space_analysis(Dataset* dataset, KMeans_Model* model) {
    print_section("ANÁLISIS DEL ESPACIO DE CARACTERÍSTICAS");
    
    if (dataset->num_features < 2) {
        print_error("Se necesitan al menos 2 características para el análisis");
        return;
    }
    
    printf("Este análisis muestra cómo se distribuyen los clusters en el espacio de características.\n");
    printf("Se analizan las dos características más importantes para la separación de clusters.\n\n");
    
    // Encontrar las características con mayor varianza entre clusters
    double feature_importance[MAX_FEATURES] = {0};
    
    for (int f = 0; f < dataset->num_features; f++) {
        // Calcular varianza total
        double total_mean = 0.0;
        for (int i = 0; i < dataset->num_points; i++) {
            total_mean += dataset->points[i].features[f];
        }
        total_mean /= dataset->num_points;
        
        double total_variance = 0.0;
        for (int i = 0; i < dataset->num_points; i++) {
            double diff = dataset->points[i].features[f] - total_mean;
            total_variance += diff * diff;
        }
        
        // Calcular varianza entre clusters
        double between_cluster_variance = 0.0;
        for (int c = 0; c < model->num_clusters; c++) {
            if (model->clusters[c].point_count > 0) {
                double cluster_mean = 0.0;
                for (int i = 0; i < model->clusters[c].point_count; i++) {
                    cluster_mean += model->clusters[c].points[i].features[f];
                }
                cluster_mean /= model->clusters[c].point_count;
                
                double diff = cluster_mean - total_mean;
                between_cluster_variance += model->clusters[c].point_count * diff * diff;
            }
        }
        
        // Calcular importancia (ratio de varianza entre/total)
        feature_importance[f] = (total_variance > 0) ? 
            between_cluster_variance / total_variance : 0;
    }
    
    // Encontrar las dos características más importantes
    int top_features[2] = {0, 1};
    for (int f = 0; f < dataset->num_features; f++) {
        if (feature_importance[f] > feature_importance[top_features[0]]) {
            top_features[1] = top_features[0];
            top_features[0] = f;
        } else if (feature_importance[f] > feature_importance[top_features[1]]) {
            top_features[1] = f;
        }
    }
    
    printf("Características más importantes para la separación de clusters:\n");
    printf("1. %s (importancia: %.3f)\n", 
           dataset->feature_names[top_features[0]], 
           feature_importance[top_features[0]]);
    printf("2. %s (importancia: %.3f)\n", 
           dataset->feature_names[top_features[1]], 
           feature_importance[top_features[1]]);
    
    // Visualización de proyección en las características principales
    printf("\n📈 Proyección en el espacio de las características principales:\n");
    
    int grid_size = 50;
    int grid_height = 25;
    char grid[grid_height][grid_size];
    
    for (int i = 0; i < grid_height; i++) {
        for (int j = 0; j < grid_size; j++) {
            grid[i][j] = ' ';
        }
    }
    
    // Calcular límites para las características principales
    double min_x = 1e10, max_x = -1e10, min_y = 1e10, max_y = -1e10;
    
    for (int i = 0; i < dataset->num_points; i++) {
        double x = dataset->points[i].features[top_features[0]];
        double y = dataset->points[i].features[top_features[1]];
        
        if (x < min_x) min_x = x;
        if (x > max_x) max_x = x;
        if (y < min_y) min_y = y;
        if (y > max_y) max_y = y;
    }
    
    // Añadir márgenes
    min_x -= (max_x - min_x) * 0.1;
    max_x += (max_x - min_x) * 0.1;
    min_y -= (max_y - min_y) * 0.1;
    max_y += (max_y - min_y) * 0.1;
    
    // Dibujar puntos
    const char cluster_symbols[] = {'*', '#', '@', '+', 'x', 'o', 's'};
    const char* cluster_colors[] = {COLOR_RED, COLOR_GREEN, COLOR_BLUE, COLOR_YELLOW, 
                                   COLOR_MAGENTA, COLOR_CYAN};
    
    for (int i = 0; i < dataset->num_points && i < 100; i++) {
        int cluster_id = dataset->points[i].cluster_id;
        
        if (cluster_id >= 0 && cluster_id < model->num_clusters) {
            int x = (int)((dataset->points[i].features[top_features[0]] - min_x) / 
                         (max_x - min_x) * (grid_size - 1));
            int y = (int)((dataset->points[i].features[top_features[1]] - min_y) / 
                         (max_y - min_y) * (grid_height - 1));
            y = grid_height - 1 - y;
            
            if (x >= 0 && x < grid_size && y >= 0 && y < grid_height) {
                grid[y][x] = cluster_symbols[cluster_id % 7];
            }
        }
    }
    
    // Dibujar centroides
    for (int c = 0; c < model->num_clusters; c++) {
        int x = (int)((model->clusters[c].centroid[top_features[0]] - min_x) / 
                     (max_x - min_x) * (grid_size - 1));
        int y = (int)((model->clusters[c].centroid[top_features[1]] - min_y) / 
                     (max_y - min_y) * (grid_height - 1));
        y = grid_height - 1 - y;
        
        if (x >= 0 && x < grid_size && y >= 0 && y < grid_height) {
            grid[y][x] = 'X';
        }
    }
    
    // Imprimir grid
    printf("\n    %s\n", dataset->feature_names[top_features[1]]);
    printf("    ↑\n");
    
    for (int i = 0; i < grid_height; i++) {
        printf("%4.1f│", max_y - (max_y - min_y) * i / (grid_height - 1));
        
        for (int j = 0; j < grid_size; j++) {
            char c = grid[i][j];
            
            if (c == 'X') {
                printf("\033[1;37mX\033[0m");
            } else if (c >= '*' && c <= 's') {
                // Encontrar cluster
                for (int k = 0; k < 7; k++) {
                    if (cluster_symbols[k] == c) {
                        int cluster_id = k % model->num_clusters;
                        printf("%s%c%s", cluster_colors[cluster_id % 6], c, COLOR_RESET);
                        break;
                    }
                }
            } else {
                printf("%c", c);
            }
        }
        printf("\n");
    }
    
    printf("    └");
    for (int j = 0; j < grid_size; j++) printf("─");
    printf("→ %s\n", dataset->feature_names[top_features[0]]);
    
    // Análisis de separación
    printf("\n🔍 Análisis de separación en este espacio:\n");
    
    // Calcular distancias entre centroides
    printf("  Distancias entre centroides:\n");
    for (int i = 0; i < model->num_clusters; i++) {
        for (int j = i + 1; j < model->num_clusters; j++) {
            double dist = euclidean_distance(model->clusters[i].centroid,
                                           model->clusters[j].centroid,
                                           dataset->num_features);
            
            printf("    Cluster %d ↔ Cluster %d: %.3f", i + 1, j + 1, dist);
            
            if (dist < 0.5) printf(" ⚠️  Muy cercanos\n");
            else if (dist < 1.0) printf(" ⚠️  Cercanos\n");
            else printf(" ✅ Bien separados\n");
        }
    }
    
    // Calcular solapamiento aproximado
    int overlapping_points = 0;
    for (int i = 0; i < dataset->num_points; i++) {
        int cluster_id = dataset->points[i].cluster_id;
        double dist_to_own = euclidean_distance(dataset->points[i].features,
                                              model->clusters[cluster_id].centroid,
                                              dataset->num_features);
        
        // Verificar si está más cerca de otro centroide
        for (int c = 0; c < model->num_clusters; c++) {
            if (c != cluster_id) {
                double dist_to_other = euclidean_distance(dataset->points[i].features,
                                                        model->clusters[c].centroid,
                                                        dataset->num_features);
                
                if (dist_to_other < dist_to_own * 1.2) {  // 20% más cercano
                    overlapping_points++;
                    break;
                }
            }
        }
    }
    
    printf("\n  Puntos en regiones de solapamiento: %d (%.1f%%)\n",
           overlapping_points,
           (double)overlapping_points / dataset->num_points * 100);
}

void print_model_info(KMeans_Model* model) {
    print_section("INFORMACIÓN COMPLETA DEL MODELO K-MEANS");
    
    printf("📋 INFORMACIÓN BÁSICA:\n");
    printf("  • Nombre: %s\n", model->name);
    printf("  • Método de inicialización: %s\n", model->initialization_method);
    printf("  • Fecha de entrenamiento: %s", ctime(&model->trained_at));
    printf("  • Tiempo de entrenamiento: %.2f segundos\n", model->training_time);
    printf("  • Iteraciones: %d\n", model->iterations);
    printf("  • Convergió: %s\n", model->converged ? "Sí" : "No");
    printf("  • Umbral de convergencia: %.6f\n", model->convergence_threshold);
    printf("  • Características usadas: %d\n", model->num_features_trained);
    
    printf("\n📊 MÉTRICAS DE CALIDAD:\n");
    printf("  • Inercia total: %.4f\n", model->total_inertia);
    printf("  • Puntuación de silueta: %.4f\n", model->silhouette_score);
    printf("  • Índice de Davies-Bouldin: %.4f\n", model->davies_bouldin_score);
    
    printf("\n🎯 INTERPRETACIÓN DE MÉTRICAS:\n");
    
    // Interpretar silueta
    if (model->silhouette_score > 0.7) {
        printf("  • Silueta: Excelente estructura de clusters (> 0.7)\n");
    } else if (model->silhouette_score > 0.5) {
        printf("  • Silueta: Buena estructura (0.5 - 0.7)\n");
    } else if (model->silhouette_score > 0.25) {
        printf("  • Silueta: Estructura débil (0.25 - 0.5)\n");
    } else if (model->silhouette_score >= 0) {
        printf("  • Silueta: Sin estructura clara (< 0.25)\n");
    } else {
        printf("  • Silueta: Valor negativo - posible problema\n");
    }
    
    // Interpretar Davies-Bouldin (menor es mejor)
    if (model->davies_bouldin_score < 0.5) {
        printf("  • Davies-Bouldin: Excelente separación (< 0.5)\n");
    } else if (model->davies_bouldin_score < 1.0) {
        printf("  • Davies-Bouldin: Buena separación (0.5 - 1.0)\n");
    } else if (model->davies_bouldin_score < 2.0) {
        printf("  • Davies-Bouldin: Separación aceptable (1.0 - 2.0)\n");
    } else {
        printf("  • Davies-Bouldin: Separación pobre (> 2.0)\n");
    }
    
    printf("\n📈 ESTADÍSTICAS DE CONVERGENCIA:\n");
    if (model->iterations > 1) {
        double inertia_start = model->inertia_history[0];
        double inertia_end = model->total_inertia;
        double improvement = (inertia_start - inertia_end) / inertia_start * 100;
        
        printf("  • Inercia inicial: %.4f\n", inertia_start);
        printf("  • Inercia final: %.4f\n", inertia_end);
        printf("  • Mejora total: %.1f%%\n", improvement);
        printf("  • Mejora por iteración: %.1f%%\n", improvement / model->iterations);
        
        // Verificar convergencia rápida
        if (model->iterations < 10 && improvement > 80) {
            printf("  • ✅ Convergencia rápida y efectiva\n");
        } else if (model->iterations > 50 && improvement < 30) {
            printf("  • ⚠️  Convergencia lenta con poca mejora\n");
        }
    }
    
    printf("\n🔍 DIAGNÓSTICO DEL MODELO:\n");
    
    // Verificar clusters vacíos
    int empty_clusters = 0;
    for (int i = 0; i < model->num_clusters; i++) {
        if (model->clusters[i].point_count == 0) {
            empty_clusters++;
        }
    }
    
    if (empty_clusters > 0) {
        printf("  • ⚠️  %d clusters están vacíos\n", empty_clusters);
        printf("    Considera: Reducir K o cambiar inicialización\n");
    } else {
        printf("  • ✅ Todos los clusters tienen puntos asignados\n");
    }
    
    // Verificar clusters desbalanceados
    int min_points = INT_MAX, max_points = 0;
    for (int i = 0; i < model->num_clusters; i++) {
        if (model->clusters[i].point_count < min_points) 
            min_points = model->clusters[i].point_count;
        if (model->clusters[i].point_count > max_points) 
            max_points = model->clusters[i].point_count;
    }
    
    double balance_ratio = (max_points > 0) ? (double)min_points / max_points : 0;
    
    if (balance_ratio < 0.1) {
        printf("  • ⚠️  Clusters muy desbalanceados (ratio: %.2f)\n", balance_ratio);
        printf("    El cluster más grande tiene %d veces más puntos\n", max_points / (min_points + 1));
    } else if (balance_ratio < 0.3) {
        printf("  • ⚠️  Clusters moderadamente desbalanceados (ratio: %.2f)\n", balance_ratio);
    } else {
        printf("  • ✅ Clusters razonablemente balanceados (ratio: %.2f)\n", balance_ratio);
    }
    
    // Verificar estabilidad de centroides
    int stable_centroids = 0;
    for (int i = 0; i < model->num_clusters; i++) {
        if (model->clusters[i].is_stable) stable_centroids++;
    }
    
    printf("  • Centroides estables: %d/%d\n", stable_centroids, model->num_clusters);
    
    printf("\n💡 RECOMENDACIONES:\n");
    
    if (model->silhouette_score < 0.3) {
        printf("  • Considera probar un valor diferente de K\n");
    }
    
    if (empty_clusters > 0) {
        printf("  • Considera usar k-means++ para inicialización\n");
    }
    
    if (strcmp(model->initialization_method, "random") == 0 && model->num_clusters > 3) {
        printf("  • Para muchos clusters, k-means++ suele dar mejores resultados\n");
    }
    
    if (model->iterations == MAX_ITERATIONS && !model->converged) {
        printf("  • El modelo no convergió en el máximo de iteraciones\n");
        printf("  • Considera aumentar MAX_ITERATIONS o el umbral de convergencia\n");
    }
}

void print_dataset_visualization(Dataset* dataset) {
    if (dataset->num_points == 0) {
        print_error("Dataset vacío");
        return;
    }
    
    print_section("VISUALIZACIÓN DEL DATASET");
    
    printf("Información general:\n");
    printf("  • Puntos: %d\n", dataset->num_points);
    printf("  • Características: %d\n", dataset->num_features);
    printf("  • Nombre: %s\n", dataset->name);
    printf("  • Normalizado: %s\n", dataset->is_normalized ? "Sí" : "No");
    
    if (strlen(dataset->description) > 0) {
        printf("  • Descripción: %s\n", dataset->description);
    }
    
    // Visualización simple si hay 2 características
    if (dataset->num_features >= 2) {
        printf("\n📈 DISTRIBUCIÓN EN 2D (primeras 2 características):\n");
        
        int grid_size = 50;
        int grid_height = 25;
        char grid[grid_height][grid_size];
        
        for (int i = 0; i < grid_height; i++) {
            for (int j = 0; j < grid_size; j++) {
                grid[i][j] = ' ';
            }
        }
        
        double min_x = dataset->feature_min[0];
        double max_x = dataset->feature_max[0];
        double min_y = dataset->feature_min[1];
        double max_y = dataset->feature_max[1];
        
        // Contar puntos por celda - usar memset para inicializar
        int density[grid_height][grid_size];
        for (int i = 0; i < grid_height; i++) {
            for (int j = 0; j < grid_size; j++) {
                density[i][j] = 0;
            }
        }
        
        for (int p = 0; p < dataset->num_points && p < 500; p++) {
            int x = (int)((dataset->points[p].features[0] - min_x) / (max_x - min_x) * (grid_size - 1));
            int y = (int)((dataset->points[p].features[1] - min_y) / (max_y - min_y) * (grid_height - 1));
            y = grid_height - 1 - y;
            
            if (x >= 0 && x < grid_size && y >= 0 && y < grid_height) {
                density[y][x]++;
            }
        }
        
        // Crear visualización con caracteres de densidad
        const char* density_chars = " .:oO@";
        
        for (int i = 0; i < grid_height; i++) {
            for (int j = 0; j < grid_size; j++) {
                int d = density[i][j];
                if (d > 10) grid[i][j] = density_chars[5];
                else if (d > 5) grid[i][j] = density_chars[4];
                else if (d > 2) grid[i][j] = density_chars[3];
                else if (d > 0) grid[i][j] = density_chars[2];
                else if (i % 5 == 0 && j % 5 == 0) grid[i][j] = density_chars[1];
                else grid[i][j] = density_chars[0];
            }
        }
        
        // Imprimir grid
        printf("    y\n");
        printf("    ↑\n");
        for (int i = 0; i < grid_height; i++) {
            printf("%4.1f│", max_y - (max_y - min_y) * i / (grid_height - 1));
            
            for (int j = 0; j < grid_size; j++) {
                char c = grid[i][j];
                if (c == '@') printf("\033[1;37m@\033[0m");
                else if (c == 'O') printf("\033[37mO\033[0m");
                else if (c == 'o') printf("\033[90mo\033[0m");
                else if (c == ':') printf("\033[90m:\033[0m");
                else if (c == '.') printf("\033[90m.\033[0m");
                else printf("%c", c);
            }
            printf("\n");
        }
        
        printf("    └");
        for (int j = 0; j < grid_size; j++) printf("─");
        printf("→ %s\n", dataset->feature_names[0]);
        
        printf("\nLeyenda de densidad:\n");
        for (int i = 0; i < 6; i++) {
            printf("  %c = ", density_chars[i]);
            if (i == 0) printf("vacío\n");
            else if (i == 1) printf("referencia\n");
            else if (i == 2) printf("1-2 puntos\n");
            else if (i == 3) printf("3-5 puntos\n");
            else if (i == 4) printf("6-10 puntos\n");
            else printf(">10 puntos\n");
        }
    }
    
    // Estadísticas por característica
    printf("\n📐 ESTADÍSTICAS POR CARACTERÍSTICA:\n");
    printf("┌─────┬──────────────────────┬────────────┬────────────┬────────────┬────────────┐\n");
    printf("│ No. │ Nombre               │   Mínimo   │   Máximo   │   Media    │  Desv.Est. │\n");
    printf("├─────┼──────────────────────┼────────────┼────────────┼────────────┼────────────┤\n");
    
    for (int i = 0; i < dataset->num_features && i < 6; i++) {
        double sum = 0.0;
        double sum_sq = 0.0;
        
        for (int j = 0; j < dataset->num_points; j++) {
            double val = dataset->points[j].features[i];
            sum += val;
            sum_sq += val * val;
        }
        
        double mean = sum / dataset->num_points;
        double variance = sum_sq / dataset->num_points - mean * mean;
        double stddev = sqrt(fmax(variance, 0));
        
        printf("│ %3d │ %-20s │ %10.4f │ %10.4f │ %10.4f │ %10.4f │\n",
               i + 1,
               dataset->feature_names[i],
               dataset->feature_min[i],
               dataset->feature_max[i],
               mean,
               stddev);
    }
    
    printf("└─────┴──────────────────────┴────────────┴────────────┴────────────┴────────────┘\n");
    
    // Análisis de estructura de clusters (estimado)
    if (dataset->num_features >= 2) {
        printf("\n🔍 ANÁLISIS PRELIMINAR DE ESTRUCTURA:\n");
        
        // Calcular densidad promedio
        double avg_density = (double)dataset->num_points / 
                            ((dataset->feature_max[0] - dataset->feature_min[0]) *
                             (dataset->feature_max[1] - dataset->feature_min[1]));
        
        printf("  • Densidad aproximada: %.2f puntos por unidad²\n", avg_density);
        
        // Estimación simple de número de clusters
        if (avg_density > 10) {
            printf("  • Alta densidad - pueden existir clusters superpuestos\n");
        } else if (avg_density > 1) {
            printf("  • Densidad media - clusters potencialmente separables\n");
        } else {
            printf("  • Baja densidad - datos dispersos\n");
        }
        
        // Verificar outliers simples
        int potential_outliers = 0;
        for (int i = 0; i < dataset->num_points; i++) {
            // Puntos muy lejos del centroide aproximado
            double center_x = (dataset->feature_max[0] + dataset->feature_min[0]) / 2;
            double center_y = (dataset->feature_max[1] + dataset->feature_min[1]) / 2;
            
            double dist = sqrt(pow(dataset->points[i].features[0] - center_x, 2) +
                              pow(dataset->points[i].features[1] - center_y, 2));
            
            double max_dist = sqrt(pow(dataset->feature_max[0] - center_x, 2) +
                                  pow(dataset->feature_max[1] - center_y, 2));
            
            if (dist > max_dist * 0.8) {
                potential_outliers++;
            }
        }
        
        printf("  • Puntos potencialmente atípicos: %d (%.1f%%)\n",
               potential_outliers,
               (double)potential_outliers / dataset->num_points * 100);
    }
    
    // Sugerencias educativas
    if (learning_mode >= 1) {
        printf("\n🎓 EJERCICIOS SUGERIDOS:\n");
        printf("1. Antes de ejecutar K-Means, intenta estimar:\n");
        printf("   • ¿Cuántos clusters naturales ves?\n");
        printf("   • ¿Hay puntos atípicos evidentes?\n");
        printf("   • ¿Los datos parecen estar agrupados?\n");
        
        printf("\n2. Preguntas para reflexionar:\n");
        printf("   • ¿Qué valor de K probarías primero?\n");
        printf("   • ¿Qué características parecen más importantes?\n");
        printf("   • ¿Necesitarías normalizar los datos?\n");
    }
}

// ============================ DATASETS ============================

Dataset create_random_clusters_dataset(int points, int clusters, double spread) {
    Dataset dataset = {0};
    dataset.num_points = points;
    dataset.num_features = 2;
    
    strcpy(dataset.feature_names[0], "Característica 1");
    strcpy(dataset.feature_names[1], "Característica 2");
    snprintf(dataset.name, sizeof(dataset.name), 
             "Dataset %d Clusters Aleatorios", clusters);
    
    // Generar centros de clusters
    double cluster_centers[MAX_CLUSTERS][2];
    for (int c = 0; c < clusters; c++) {
        cluster_centers[c][0] = random_double(-5.0, 5.0);
        cluster_centers[c][1] = random_double(-5.0, 5.0);
    }
    
    // Generar puntos alrededor de los centros
    int points_per_cluster = points / clusters;
    int point_index = 0;
    
    for (int c = 0; c < clusters && point_index < points; c++) {
        for (int p = 0; p < points_per_cluster && point_index < points; p++) {
            dataset.points[point_index].features[0] = 
                cluster_centers[c][0] + random_double(-spread, spread);
            dataset.points[point_index].features[1] = 
                cluster_centers[c][1] + random_double(-spread, spread);
            
            // Actualizar min/max
            if (point_index == 0) {
                dataset.feature_min[0] = dataset.feature_max[0] = 
                    dataset.points[point_index].features[0];
                dataset.feature_min[1] = dataset.feature_max[1] = 
                    dataset.points[point_index].features[1];
            } else {
                if (dataset.points[point_index].features[0] < dataset.feature_min[0])
                    dataset.feature_min[0] = dataset.points[point_index].features[0];
                if (dataset.points[point_index].features[0] > dataset.feature_max[0])
                    dataset.feature_max[0] = dataset.points[point_index].features[0];
                if (dataset.points[point_index].features[1] < dataset.feature_min[1])
                    dataset.feature_min[1] = dataset.points[point_index].features[1];
                if (dataset.points[point_index].features[1] > dataset.feature_max[1])
                    dataset.feature_max[1] = dataset.points[point_index].features[1];
            }
            
            point_index++;
        }
    }
    
    // Añadir puntos restantes aleatorios
    while (point_index < points) {
        dataset.points[point_index].features[0] = random_double(-8.0, 8.0);
        dataset.points[point_index].features[1] = random_double(-8.0, 8.0);
        point_index++;
    }
    
    return dataset;
}

Dataset create_circular_clusters_dataset(int points, int circles) {
    Dataset dataset = {0};
    dataset.num_points = points;
    dataset.num_features = 2;
    
    strcpy(dataset.feature_names[0], "X");
    strcpy(dataset.feature_names[1], "Y");
    snprintf(dataset.name, sizeof(dataset.name),
             "Dataset %d Círculos Concéntricos", circles);
    strcpy(dataset.description, "Clusters circulares concéntricos, desafiantes para K-Means");
    
    // Crear círculos concéntricos
    int points_per_circle = points / circles;
    int point_index = 0;
    
    for (int circle = 1; circle <= circles && point_index < points; circle++) {
        double radius = circle * 1.5;
        
        for (int p = 0; p < points_per_circle && point_index < points; p++) {
            double angle = random_double(0, 2 * M_PI);
            double r = radius + random_double(-0.3, 0.3);
            
            dataset.points[point_index].features[0] = r * cos(angle);
            dataset.points[point_index].features[1] = r * sin(angle);
            
            // Actualizar min/max
            if (point_index == 0) {
                dataset.feature_min[0] = dataset.feature_max[0] = 
                    dataset.points[point_index].features[0];
                dataset.feature_min[1] = dataset.feature_max[1] = 
                    dataset.points[point_index].features[1];
            } else {
                if (dataset.points[point_index].features[0] < dataset.feature_min[0])
                    dataset.feature_min[0] = dataset.points[point_index].features[0];
                if (dataset.points[point_index].features[0] > dataset.feature_max[0])
                    dataset.feature_max[0] = dataset.points[point_index].features[0];
                if (dataset.points[point_index].features[1] < dataset.feature_min[1])
                    dataset.feature_min[1] = dataset.points[point_index].features[1];
                if (dataset.points[point_index].features[1] > dataset.feature_max[1])
                    dataset.feature_max[1] = dataset.points[point_index].features[1];
            }
            
            point_index++;
        }
    }
    
    return dataset;
}

Dataset create_moon_shaped_dataset(int points, int moons) {
    Dataset dataset = {0};
    dataset.num_points = points;
    dataset.num_features = 2;
    
    strcpy(dataset.feature_names[0], "X");
    strcpy(dataset.feature_names[1], "Y");
    snprintf(dataset.name, sizeof(dataset.name),
             "Dataset %d Medias Lunas", moons);
    strcpy(dataset.description, "Datos en forma de medias lunas entrelazadas");
    
    int points_per_moon = points / moons;
    int point_index = 0;
    
    for (int moon = 0; moon < moons && point_index < points; moon++) {
        double angle_offset = moon * M_PI;
        double x_offset = moon * 3.0;
        
        for (int p = 0; p < points_per_moon && point_index < points; p++) {
            double angle = random_double(0, M_PI);
            double r = 1.0 + random_double(-0.2, 0.2);
            
            dataset.points[point_index].features[0] = 
                r * cos(angle + angle_offset) + x_offset + random_double(-0.1, 0.1);
            dataset.points[point_index].features[1] = 
                r * sin(angle) + random_double(-0.1, 0.1);
            
            // Actualizar min/max
            if (point_index == 0) {
                dataset.feature_min[0] = dataset.feature_max[0] = 
                    dataset.points[point_index].features[0];
                dataset.feature_min[1] = dataset.feature_max[1] = 
                    dataset.points[point_index].features[1];
            } else {
                if (dataset.points[point_index].features[0] < dataset.feature_min[0])
                    dataset.feature_min[0] = dataset.points[point_index].features[0];
                if (dataset.points[point_index].features[0] > dataset.feature_max[0])
                    dataset.feature_max[0] = dataset.points[point_index].features[0];
                if (dataset.points[point_index].features[1] < dataset.feature_min[1])
                    dataset.feature_min[1] = dataset.points[point_index].features[1];
                if (dataset.points[point_index].features[1] > dataset.feature_max[1])
                    dataset.feature_max[1] = dataset.points[point_index].features[1];
            }
            
            point_index++;
        }
    }
    
    return dataset;
}

Dataset create_spiral_dataset(int points, int arms) {
    Dataset dataset = {0};
    dataset.num_points = points;
    dataset.num_features = 2;
    
    strcpy(dataset.feature_names[0], "X");
    strcpy(dataset.feature_names[1], "Y");
    snprintf(dataset.name, sizeof(dataset.name),
             "Dataset %d Espirales", arms);
    strcpy(dataset.description, "Datos en forma de espirales entrelazadas");
    
    int points_per_arm = points / arms;
    int point_index = 0;
    
    for (int arm = 0; arm < arms && point_index < points; arm++) {
        double angle_offset = arm * 2 * M_PI / arms;
        
        for (int p = 0; p < points_per_arm && point_index < points; p++) {
            double t = (double)p / points_per_arm * 4 * M_PI;
            double r = t / (4 * M_PI) * 5.0;
            
            dataset.points[point_index].features[0] = 
                r * cos(t + angle_offset) + random_double(-0.1, 0.1);
            dataset.points[point_index].features[1] = 
                r * sin(t + angle_offset) + random_double(-0.1, 0.1);
            
            // Actualizar min/max
            if (point_index == 0) {
                dataset.feature_min[0] = dataset.feature_max[0] = 
                    dataset.points[point_index].features[0];
                dataset.feature_min[1] = dataset.feature_max[1] = 
                    dataset.points[point_index].features[1];
            } else {
                if (dataset.points[point_index].features[0] < dataset.feature_min[0])
                    dataset.feature_min[0] = dataset.points[point_index].features[0];
                if (dataset.points[point_index].features[0] > dataset.feature_max[0])
                    dataset.feature_max[0] = dataset.points[point_index].features[0];
                if (dataset.points[point_index].features[1] < dataset.feature_min[1])
                    dataset.feature_min[1] = dataset.points[point_index].features[1];
                if (dataset.points[point_index].features[1] > dataset.feature_max[1])
                    dataset.feature_max[1] = dataset.points[point_index].features[1];
            }
            
            point_index++;
        }
    }
    
    return dataset;
}

Dataset create_blob_dataset(int points, int blobs, double cluster_std) {
    Dataset dataset = {0};
    dataset.num_points = points;
    dataset.num_features = 2;
    
    strcpy(dataset.feature_names[0], "X");
    strcpy(dataset.feature_names[1], "Y");
    snprintf(dataset.name, sizeof(dataset.name),
             "Dataset %d Blobs", blobs);
    strcpy(dataset.description, "Datos en forma de blobs Gaussianos");
    
    // Generar centros de blobs
    double blob_centers[MAX_CLUSTERS][2];
    for (int b = 0; b < blobs; b++) {
        blob_centers[b][0] = random_double(-5.0, 5.0);
        blob_centers[b][1] = random_double(-5.0, 5.0);
    }
    
    int points_per_blob = points / blobs;
    int point_index = 0;
    
    for (int b = 0; b < blobs && point_index < points; b++) {
        for (int p = 0; p < points_per_blob && point_index < points; p++) {
            dataset.points[point_index].features[0] = 
                blob_centers[b][0] + random_double(-cluster_std, cluster_std);
            dataset.points[point_index].features[1] = 
                blob_centers[b][1] + random_double(-cluster_std, cluster_std);
            
            // Actualizar min/max
            if (point_index == 0) {
                dataset.feature_min[0] = dataset.feature_max[0] = 
                    dataset.points[point_index].features[0];
                dataset.feature_min[1] = dataset.feature_max[1] = 
                    dataset.points[point_index].features[1];
            } else {
                if (dataset.points[point_index].features[0] < dataset.feature_min[0])
                    dataset.feature_min[0] = dataset.points[point_index].features[0];
                if (dataset.points[point_index].features[0] > dataset.feature_max[0])
                    dataset.feature_max[0] = dataset.points[point_index].features[0];
                if (dataset.points[point_index].features[1] < dataset.feature_min[1])
                    dataset.feature_min[1] = dataset.points[point_index].features[1];
                if (dataset.points[point_index].features[1] > dataset.feature_max[1])
                    dataset.feature_max[1] = dataset.points[point_index].features[1];
            }
            
            point_index++;
        }
    }
    
    return dataset;
}

Dataset create_anisotropic_dataset(int points) {
    Dataset dataset = {0};
    dataset.num_points = points;
    dataset.num_features = 2;
    
    strcpy(dataset.feature_names[0], "X");
    strcpy(dataset.feature_names[1], "Y");
    strcpy(dataset.name, "Dataset Anisotrópico");
    strcpy(dataset.description, "Datos con varianza diferente en cada dirección");
    
    // Crear clusters anisotrópicos
    for (int i = 0; i < points; i++) {
        if (i < points / 3) {
            // Cluster 1: varianza mayor en X
            dataset.points[i].features[0] = random_double(-2.0, 2.0) + 3.0;
            dataset.points[i].features[1] = random_double(-0.5, 0.5);
        } else if (i < 2 * points / 3) {
            // Cluster 2: varianza mayor en Y
            dataset.points[i].features[0] = random_double(-0.5, 0.5) - 3.0;
            dataset.points[i].features[1] = random_double(-2.0, 2.0);
        } else {
            // Cluster 3: varianza similar en ambas direcciones
            dataset.points[i].features[0] = random_double(-1.5, 1.5);
            dataset.points[i].features[1] = random_double(-1.5, 1.5) + 4.0;
        }
        
        // Actualizar min/max
        if (i == 0) {
            dataset.feature_min[0] = dataset.feature_max[0] = 
                dataset.points[i].features[0];
            dataset.feature_min[1] = dataset.feature_max[1] = 
                dataset.points[i].features[1];
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

Dataset create_varied_variance_dataset(int points) {
    Dataset dataset = {0};
    dataset.num_points = points;
    dataset.num_features = 2;
    
    strcpy(dataset.feature_names[0], "X");
    strcpy(dataset.feature_names[1], "Y");
    strcpy(dataset.name, "Dataset con Varianza Variable");
    strcpy(dataset.description, "Clusters con diferentes niveles de dispersión");
    
    int clusters = 3;
    int points_per_cluster = points / clusters;
    int point_index = 0;
    
    double cluster_centers[3][2] = {
        {-4.0, 0.0},
        {0.0, 0.0},
        {4.0, 0.0}
    };
    
    double cluster_std[3] = {0.3, 1.0, 2.0};
    
    for (int c = 0; c < clusters && point_index < points; c++) {
        for (int p = 0; p < points_per_cluster && point_index < points; p++) {
            dataset.points[point_index].features[0] = 
                cluster_centers[c][0] + random_double(-cluster_std[c], cluster_std[c]);
            dataset.points[point_index].features[1] = 
                cluster_centers[c][1] + random_double(-cluster_std[c], cluster_std[c]);
            
            // Actualizar min/max
            if (point_index == 0) {
                dataset.feature_min[0] = dataset.feature_max[0] = 
                    dataset.points[point_index].features[0];
                dataset.feature_min[1] = dataset.feature_max[1] = 
                    dataset.points[point_index].features[1];
            } else {
                if (dataset.points[point_index].features[0] < dataset.feature_min[0])
                    dataset.feature_min[0] = dataset.points[point_index].features[0];
                if (dataset.points[point_index].features[0] > dataset.feature_max[0])
                    dataset.feature_max[0] = dataset.points[point_index].features[0];
                if (dataset.points[point_index].features[1] < dataset.feature_min[1])
                    dataset.feature_min[1] = dataset.points[point_index].features[1];
                if (dataset.points[point_index].features[1] > dataset.feature_max[1])
                    dataset.feature_max[1] = dataset.points[point_index].features[1];
            }
            
            point_index++;
        }
    }
    
    return dataset;
}

Dataset create_no_structure_dataset(int points) {
    Dataset dataset = {0};
    dataset.num_points = points;
    dataset.num_features = 2;
    
    strcpy(dataset.feature_names[0], "X");
    strcpy(dataset.feature_names[1], "Y");
    strcpy(dataset.name, "Dataset sin Estructura");
    strcpy(dataset.description, "Datos uniformemente distribuidos, sin clusters naturales");
    
    for (int i = 0; i < points; i++) {
        dataset.points[i].features[0] = random_double(-5.0, 5.0);
        dataset.points[i].features[1] = random_double(-5.0, 5.0);
        
        // Actualizar min/max
        if (i == 0) {
            dataset.feature_min[0] = dataset.feature_max[0] = 
                dataset.points[i].features[0];
            dataset.feature_min[1] = dataset.feature_max[1] = 
                dataset.points[i].features[1];
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
    if (dataset->is_normalized || dataset->num_points == 0) {
        return;
    }
    
    for (int i = 0; i < dataset->num_points; i++) {
        for (int j = 0; j < dataset->num_features; j++) {
            // Normalización min-max a [0, 1]
            if (dataset->feature_max[j] - dataset->feature_min[j] > 0) {
                dataset->points[i].features[j] = 
                    (dataset->points[i].features[j] - dataset->feature_min[j]) /
                    (dataset->feature_max[j] - dataset->feature_min[j]);
            }
        }
    }
    
    // Actualizar min/max después de normalizar
    for (int j = 0; j < dataset->num_features; j++) {
        dataset->feature_min[j] = 0.0;
        dataset->feature_max[j] = 1.0;
    }
    
    dataset->is_normalized = 1;
}

void print_dataset_info(Dataset* dataset) {
    if (dataset->num_points == 0) {
        print_error("Dataset vacío");
        return;
    }
    
    print_section("INFORMACIÓN DEL DATASET");
    
    printf("📋 Información básica:\n");
    printf("  • Nombre: %s\n", dataset->name);
    printf("  • Puntos: %d\n", dataset->num_points);
    printf("  • Características: %d\n", dataset->num_features);
    printf("  • Normalizado: %s\n", dataset->is_normalized ? "Sí" : "No");
    
    if (strlen(dataset->description) > 0) {
        printf("  • Descripción: %s\n", dataset->description);
    }
    
    // Rango de características
    printf("\n📐 Rango de características:\n");
    for (int i = 0; i < dataset->num_features && i < 5; i++) {
        printf("  • %s: [%.4f, %.4f]\n", 
               dataset->feature_names[i],
               dataset->feature_min[i],
               dataset->feature_max[i]);
    }
    
    if (dataset->num_features > 5) {
        printf("  • ... y %d características más\n", dataset->num_features - 5);
    }
}

// ============================ FUNCIONES FALTANTES ============================

// Función save_dataset que estaba faltando
void save_dataset(Dataset* dataset, const char* filename) {
    if (dataset->num_points == 0) {
        print_error("Dataset vacío, no hay nada que guardar");
        return;
    }

    FILE* file = fopen(filename, "w");
    if (!file) {
        print_error("No se pudo abrir el archivo para guardar");
        return;
    }

    // Escribir encabezado con nombres de características
    for (int i = 0; i < dataset->num_features; i++) {
        fprintf(file, "%s", dataset->feature_names[i]);
        if (i < dataset->num_features - 1) {
            fprintf(file, ",");
        }
    }
    fprintf(file, "\n");

    // Escribir datos
    for (int i = 0; i < dataset->num_points; i++) {
        for (int j = 0; j < dataset->num_features; j++) {
            fprintf(file, "%.6f", dataset->points[i].features[j]);
            if (j < dataset->num_features - 1) {
                fprintf(file, ",");
            }
        }
        fprintf(file, "\n");
    }

    fclose(file);
    print_success("Dataset guardado exitosamente en %s", filename);
}

// Implementación de load_dataset que faltaba
Dataset load_dataset(const char* filename) {
    Dataset dataset = {0};
    
    FILE* file = fopen(filename, "r");
    if (!file) {
        print_error("No se pudo abrir el archivo: %s", filename);
        return dataset;
    }
    
    char line[1024];
    int line_number = 0;
    
    // Leer nombres de características de la primera línea
    if (fgets(line, sizeof(line), file)) {
        line_number++;
        char* token = strtok(line, ",\n");
        int feature_count = 0;
        
        while (token != NULL && feature_count < MAX_FEATURES) {
            strncpy(dataset.feature_names[feature_count], token, 49);
            dataset.feature_names[feature_count][49] = '\0';
            token = strtok(NULL, ",\n");
            feature_count++;
        }
        
        dataset.num_features = feature_count;
    }
    
    // Leer datos
    while (fgets(line, sizeof(line), file) && dataset.num_points < MAX_POINTS) {
        line_number++;
        
        char* token = strtok(line, ",\n");
        int feature_index = 0;
        
        while (token != NULL && feature_index < dataset.num_features) {
            double value = atof(token);
            dataset.points[dataset.num_points].features[feature_index] = value;
            
            // Actualizar min/max
            if (dataset.num_points == 0) {
                dataset.feature_min[feature_index] = value;
                dataset.feature_max[feature_index] = value;
            } else {
                if (value < dataset.feature_min[feature_index])
                    dataset.feature_min[feature_index] = value;
                if (value > dataset.feature_max[feature_index])
                    dataset.feature_max[feature_index] = value;
            }
            
            token = strtok(NULL, ",\n");
            feature_index++;
        }
        
        dataset.num_points++;
    }
    
    fclose(file);
    
    // Asignar nombre basado en el nombre del archivo
    if (strlen(dataset.name) == 0) {
        strncpy(dataset.name, filename, 99);
        dataset.name[99] = '\0';
    }
    
    print_success("Dataset cargado: %d puntos, %d características", 
                  dataset.num_points, dataset.num_features);
    
    return dataset;
}

// Función calculate_davies_bouldin_score que estaba faltando
double calculate_davies_bouldin_score(Dataset* dataset, KMeans_Model* model) {
    if (model->num_clusters <= 1) {
        return 0.0;  // No se puede calcular para un solo cluster
    }
    
    double db_index = 0.0;
    double epsilon = 1e-10;  // Para evitar división por cero
    
    // Calcular para cada cluster i
    for (int i = 0; i < model->num_clusters; i++) {
        // Calcular dispersión promedio del cluster i (s_i)
        double s_i = 0.0;
        if (model->clusters[i].point_count > 0) {
            for (int p = 0; p < model->clusters[i].point_count; p++) {
                s_i += euclidean_distance(model->clusters[i].points[p].features,
                                        model->clusters[i].centroid,
                                        model->num_features_trained);
            }
            s_i /= model->clusters[i].point_count;
        }
        
        double max_ratio = -1.0;
        
        // Comparar con todos los otros clusters j
        for (int j = 0; j < model->num_clusters; j++) {
            if (i != j) {
                // Calcular dispersión promedio del cluster j (s_j)
                double s_j = 0.0;
                if (model->clusters[j].point_count > 0) {
                    for (int p = 0; p < model->clusters[j].point_count; p++) {
                        s_j += euclidean_distance(model->clusters[j].points[p].features,
                                                model->clusters[j].centroid,
                                                model->num_features_trained);
                    }
                    s_j /= model->clusters[j].point_count;
                }
                
                // Calcular distancia entre centroides i y j
                double d_ij = euclidean_distance(model->clusters[i].centroid,
                                               model->clusters[j].centroid,
                                               model->num_features_trained);
                
                // Calcular ratio (s_i + s_j) / d_ij
                if (d_ij > epsilon) {
                    double ratio = (s_i + s_j) / d_ij;
                    if (ratio > max_ratio) {
                        max_ratio = ratio;
                    }
                }
            }
        }
        
        if (max_ratio > 0) {
            db_index += max_ratio;
        }
    }
    
    // Promedio sobre todos los clusters
    return db_index / model->num_clusters;
}

// Otras funciones faltantes (implementaciones básicas)
void add_noise_to_dataset(Dataset* dataset, double noise_level) {
    if (dataset->num_points == 0) return;
    
    for (int i = 0; i < dataset->num_points; i++) {
        for (int j = 0; j < dataset->num_features; j++) {
            double range = dataset->feature_max[j] - dataset->feature_min[j];
            dataset->points[i].features[j] += random_double(-noise_level * range, noise_level * range);
        }
    }
}

void shuffle_dataset(Dataset* dataset) {
    if (dataset->num_points == 0) return;
    
    for (int i = dataset->num_points - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        // Intercambiar puntos i y j
        DataPoint temp = dataset->points[i];
        dataset->points[i] = dataset->points[j];
        dataset->points[j] = temp;
    }
}

void initialize_centroids_manual(Dataset* dataset, KMeans_Model* model) {
    // Implementación básica - similar a random
    printf("Inicialización manual no implementada completamente.\n");
    printf("Usando inicialización aleatoria en su lugar.\n");
    initialize_centroids_random(dataset, model);
    strcpy(model->initialization_method, "manual");
}

void find_optimal_k_silhouette(Dataset* dataset, int k_min, int k_max) {
    print_section("BÚSQUEDA DE K ÓPTIMO (MÉTODO DE SILUETA)");
    
    printf("Ejecutando K-Means para K desde %d hasta %d...\n", k_min, k_max);
    printf("Calculando puntuación de silueta para cada K...\n\n");
    
    KMeans_Model models[MAX_CLUSTERS];
    int num_models = k_max - k_min + 1;
    
    for (int k = k_min; k <= k_max; k++) {
        printf("Probando K = %d... ", k);
        
        // Crear y entrenar modelo
        KMeans_Model model = {0};
        model.num_clusters = k;
        strcpy(model.initialization_method, "k-means++");
        
        train_kmeans(dataset, &model, 100);
        
        // Guardar modelo
        models[k - k_min] = model;
        
        printf("Silueta: %.4f\n", model.silhouette_score);
    }
    
    // Encontrar K con mejor silueta
    double best_silhouette = -1.0;
    int best_k = k_min;
    
    for (int i = 0; i < num_models; i++) {
        if (models[i].silhouette_score > best_silhouette) {
            best_silhouette = models[i].silhouette_score;
            best_k = k_min + i;
        }
    }
    
    printf("\n🎯 K óptimo basado en silueta: %d\n", best_k);
    printf("  • Puntuación de silueta: %.4f\n", best_silhouette);
    
    // Interpretación
    printf("\n📊 Interpretación de la silueta:\n");
    if (best_silhouette > 0.7) {
        printf("  • Excelente estructura de clusters\n");
    } else if (best_silhouette > 0.5) {
        printf("  • Buena estructura\n");
    } else if (best_silhouette > 0.25) {
        printf("  • Estructura débil\n");
    } else {
        printf("  • Sin estructura clara\n");
    }
    
    // Preguntar si usar este K
    printf("\n¿Deseas entrenar un modelo con K = %d? (s/n): ", best_k);
    
    char respuesta;
    scanf("%c", &respuesta);
    getchar();
    
    if (respuesta == 's' || respuesta == 'S') {
        current_model = models[best_k - k_min];
        print_success("Modelo con K óptimo cargado");
    }
    
    wait_for_enter();
}

ClusterMetrics evaluate_clustering(KMeans_Model* model, Dataset* dataset) {
    ClusterMetrics metrics = {0};
    
    // Calcular métricas básicas
    metrics.inertia = model->total_inertia;
    metrics.silhouette_score = model->silhouette_score;
    metrics.davies_bouldin = model->davies_bouldin_score;
    
    // Calcular tamaños de clusters
    for (int i = 0; i < model->num_clusters; i++) {
        metrics.cluster_sizes[i] = model->clusters[i].point_count;
        
        // Calcular densidad aproximada
        if (model->clusters[i].radius > 0) {
            metrics.cluster_density[i] = model->clusters[i].point_count / 
                                        (M_PI * model->clusters[i].radius * model->clusters[i].radius);
        }
    }
    
    return metrics;
}

void compare_clustering_algorithms(Dataset* dataset) {
    print_section("COMPARACIÓN DE ALGORITMOS DE CLUSTERING");
    
    printf("Esta función compararía K-Means con otros algoritmos,\n");
    printf("pero actualmente solo K-Means está implementado.\n\n");
    
    printf("Algoritmos que podrían añadirse:\n");
    printf("  1. DBSCAN - Basado en densidad\n");
    printf("  2. Mean-Shift - Basado en moda\n");
    printf("  3. Hierarchical - Aglomerativo/Divisivo\n");
    printf("  4. Gaussian Mixture Models (GMM)\n");
    
    printf("\n💡 Idea para implementación futura:\n");
    printf("  Cada algoritmo tendría su propia estructura y funciones,\n");
    printf("  pero compartiría las mismas visualizaciones y métricas.\n");
    
    wait_for_enter();
}

// ============================ FUNCIONES K-MEANS ============================

double euclidean_distance(double a[], double b[], int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

int assign_points_to_clusters(Dataset* dataset, KMeans_Model* model) {
    int points_moved = 0;
    
    for (int i = 0; i < dataset->num_points; i++) {
        double min_dist = DBL_MAX;
        int closest_cluster = -1;
        
        // Encontrar el centroide más cercano
        for (int c = 0; c < model->num_clusters; c++) {
            double dist = euclidean_distance(dataset->points[i].features,
                                           model->clusters[c].centroid,
                                           model->num_features_trained);
            
            if (dist < min_dist) {
                min_dist = dist;
                closest_cluster = c;
            }
        }
        
        // Asignar punto al cluster más cercano
        if (closest_cluster != dataset->points[i].cluster_id) {
            points_moved++;
            dataset->points[i].cluster_id = closest_cluster;
        }
        
        dataset->points[i].distance_to_centroid = min_dist;
    }
    
    return points_moved;
}

int update_centroids(Dataset* dataset, KMeans_Model* model) {
    int centroids_moved = 0;
    
    for (int c = 0; c < model->num_clusters; c++) {
        // Guardar centroide anterior
        for (int f = 0; f < model->num_features_trained; f++) {
            model->clusters[c].prev_centroid[f] = model->clusters[c].centroid[f];
        }
        
        // Inicializar suma para nuevo centroide
        double sum[MAX_FEATURES] = {0};
        int point_count = 0;
        
        // Sumar todos los puntos del cluster
        for (int i = 0; i < dataset->num_points; i++) {
            if (dataset->points[i].cluster_id == c) {
                for (int f = 0; f < model->num_features_trained; f++) {
                    sum[f] += dataset->points[i].features[f];
                }
                point_count++;
            }
        }
        
        // Calcular nuevo centroide (promedio)
        if (point_count > 0) {
            for (int f = 0; f < model->num_features_trained; f++) {
                model->clusters[c].centroid[f] = sum[f] / point_count;
            }
            
            // Verificar si el centroide se movió
            double movement = euclidean_distance(model->clusters[c].centroid,
                                               model->clusters[c].prev_centroid,
                                               model->num_features_trained);
            
            if (movement > model->convergence_threshold) {
                centroids_moved++;
                model->clusters[c].is_stable = 0;
            } else {
                model->clusters[c].is_stable = 1;
            }
            
            // Actualizar información del cluster
            model->clusters[c].point_count = point_count;
        } else {
            // Cluster vacío - reinicializar aleatoriamente
            for (int f = 0; f < model->num_features_trained; f++) {
                model->clusters[c].centroid[f] = random_double(0.0, 1.0);
            }
            model->clusters[c].point_count = 0;
            model->clusters[c].is_stable = 0;
        }
    }
    
    return centroids_moved;
}

void initialize_centroids_random(Dataset* dataset, KMeans_Model* model) {
    // Seleccionar K puntos aleatorios como centroides iniciales
    for (int c = 0; c < model->num_clusters; c++) {
        int random_point = rand() % dataset->num_points;
        for (int f = 0; f < model->num_features_trained; f++) {
            model->clusters[c].centroid[f] = dataset->points[random_point].features[f];
            model->clusters[c].prev_centroid[f] = model->clusters[c].centroid[f];
        }
        model->clusters[c].id = c;
        model->clusters[c].point_count = 0;
        model->clusters[c].is_stable = 0;
        
        // Asignar color y símbolo
        const char* colors[] = {COLOR_RED, COLOR_GREEN, COLOR_BLUE, COLOR_YELLOW, 
                               COLOR_MAGENTA, COLOR_CYAN, COLOR_WHITE};
        const char symbols[] = {'*', '#', '@', '+', 'x', 'o', 's'};
        
        strcpy(model->clusters[c].color_code, colors[c % 7]);
        model->clusters[c].symbol = symbols[c % 7];
    }
    
    strcpy(model->initialization_method, "random");
}

void initialize_centroids_kmeansplusplus(Dataset* dataset, KMeans_Model* model) {
    // Paso 1: Elegir primer centroide aleatoriamente
    int first_centroid = rand() % dataset->num_points;
    for (int f = 0; f < model->num_features_trained; f++) {
        model->clusters[0].centroid[f] = dataset->points[first_centroid].features[f];
    }
    
    // Paso 2: Para los centros restantes, usar probabilidad proporcional a D(x)²
    for (int c = 1; c < model->num_clusters; c++) {
        // Calcular distancias al centroide más cercano para cada punto
        double distances[MAX_POINTS] = {0};
        double total_distance_sq = 0.0;
        
        for (int i = 0; i < dataset->num_points; i++) {
            double min_dist = DBL_MAX;
            
            // Encontrar distancia al centroide más cercano
            for (int j = 0; j < c; j++) {
                double dist = euclidean_distance(dataset->points[i].features,
                                               model->clusters[j].centroid,
                                               model->num_features_trained);
                if (dist < min_dist) min_dist = dist;
            }
            
            distances[i] = min_dist;
            total_distance_sq += min_dist * min_dist;
        }
        
        // Seleccionar próximo centroide con probabilidad proporcional a D(x)²
        double threshold = random_double(0, total_distance_sq);
        double cumulative = 0.0;
        int selected_point = -1;
        
        for (int i = 0; i < dataset->num_points; i++) {
            cumulative += distances[i] * distances[i];
            if (cumulative >= threshold) {
                selected_point = i;
                break;
            }
        }
        
        // Asegurar que seleccionamos un punto
        if (selected_point == -1) selected_point = rand() % dataset->num_points;
        
        // Asignar nuevo centroide
        for (int f = 0; f < model->num_features_trained; f++) {
            model->clusters[c].centroid[f] = dataset->points[selected_point].features[f];
            model->clusters[c].prev_centroid[f] = model->clusters[c].centroid[f];
        }
        
        model->clusters[c].id = c;
        model->clusters[c].point_count = 0;
        model->clusters[c].is_stable = 0;
        
        // Asignar color y símbolo
        const char* colors[] = {COLOR_RED, COLOR_GREEN, COLOR_BLUE, COLOR_YELLOW, 
                               COLOR_MAGENTA, COLOR_CYAN, COLOR_WHITE};  // Corregido: COLOR_MAGENTA
        const char symbols[] = {'*', '#', '@', '+', 'x', 'o', 's'};
        
        strcpy(model->clusters[c].color_code, colors[c % 7]);
        model->clusters[c].symbol = symbols[c % 7];
    }
    
    // También inicializar el primer cluster
    model->clusters[0].id = 0;
    model->clusters[0].point_count = 0;
    model->clusters[0].is_stable = 0;
    strcpy(model->clusters[0].color_code, COLOR_RED);
    model->clusters[0].symbol = '*';
    
    strcpy(model->initialization_method, "k-means++");
}

int kmeans_has_converged(KMeans_Model* model, double threshold) {
    for (int c = 0; c < model->num_clusters; c++) {
        double movement = euclidean_distance(model->clusters[c].centroid,
                                           model->clusters[c].prev_centroid,
                                           model->num_features_trained);
        
        if (movement > threshold) {
            return 0;  // No convergió
        }
    }
    return 1;  // Convergió
}

double calculate_inertia(Dataset* dataset, KMeans_Model* model) {
    double total_inertia = 0.0;
    
    for (int i = 0; i < dataset->num_points; i++) {
        int cluster_id = dataset->points[i].cluster_id;
        if (cluster_id >= 0 && cluster_id < model->num_clusters) {
            double dist = euclidean_distance(dataset->points[i].features,
                                           model->clusters[cluster_id].centroid,
                                           model->num_features_trained);
            total_inertia += dist * dist;
        }
    }
    
    return total_inertia;
}

double calculate_silhouette_score(Dataset* dataset, KMeans_Model* model) {
    if (model->num_clusters < 2) return 0.0;
    
    double total_silhouette = 0.0;
    int valid_points = 0;
    
    for (int i = 0; i < dataset->num_points; i++) {
        int cluster_i = dataset->points[i].cluster_id;
        
        // Calcular a(i): distancia promedio a puntos en el mismo cluster
        double a_i = 0.0;
        int count_a = 0;
        
        for (int j = 0; j < dataset->num_points; j++) {
            if (i != j && dataset->points[j].cluster_id == cluster_i) {
                a_i += euclidean_distance(dataset->points[i].features,
                                        dataset->points[j].features,
                                        dataset->num_features);
                count_a++;
            }
        }
        
        if (count_a == 0) continue;  // Cluster con un solo punto
        
        a_i /= count_a;
        
        // Calcular b(i): distancia mínima promedio a otros clusters
        double b_i = DBL_MAX;
        
        for (int c = 0; c < model->num_clusters; c++) {
            if (c != cluster_i) {
                double avg_dist = 0.0;
                int count_b = 0;
                
                for (int j = 0; j < dataset->num_points; j++) {
                    if (dataset->points[j].cluster_id == c) {
                        avg_dist += euclidean_distance(dataset->points[i].features,
                                                     dataset->points[j].features,
                                                     dataset->num_features);
                        count_b++;
                    }
                }
                
                if (count_b > 0) {
                    avg_dist /= count_b;
                    if (avg_dist < b_i) b_i = avg_dist;
                }
            }
        }
        
        if (b_i == DBL_MAX) continue;  // Solo hay un cluster
        
        // Calcular silueta para este punto
        double s_i = (b_i - a_i) / fmax(a_i, b_i);
        total_silhouette += s_i;
        valid_points++;
    }
    
    return (valid_points > 0) ? total_silhouette / valid_points : 0.0;
}

// ============================ ENTRENAMIENTO ============================

void train_kmeans(Dataset* dataset, KMeans_Model* model, int max_iterations) {
    print_section("ENTRENANDO K-MEANS");
    
    // Inicializar modelo
    model->num_features_trained = dataset->num_features;
    model->convergence_threshold = 0.0001;
    model->converged = 0;
    model->training_time = 0.0;
    
    // Elegir método de inicialización
    printf("Seleccionando centroides iniciales...\n");
    if (strcmp(model->initialization_method, "k-means++") == 0) {
        initialize_centroids_kmeansplusplus(dataset, model);
        printf("  • Método: k-means++\n");
    } else {
        initialize_centroids_random(dataset, model);
        printf("  • Método: aleatorio\n");
    }
    
    clock_t start_time = clock();
    
    // Iteraciones del algoritmo
    for (int iter = 0; iter < max_iterations; iter++) {
        model->iterations = iter + 1;
        
        // Paso 1: Asignar puntos a clusters
        int points_moved = assign_points_to_clusters(dataset, model);
        training_history.point_movements_history[iter] = points_moved;
        
        // Paso 2: Actualizar centroides
        int centroids_moved = update_centroids(dataset, model);
        
        // Calcular movimiento promedio de centroides
        double total_movement = 0.0;
        for (int c = 0; c < model->num_clusters; c++) {
            total_movement += euclidean_distance(model->clusters[c].centroid,
                                               model->clusters[c].prev_centroid,
                                               model->num_features_trained);
        }
        training_history.centroids_movement[iter] = total_movement / model->num_clusters;
        
        // Calcular inercia
        model->total_inertia = calculate_inertia(dataset, model);
        training_history.inertia_history[iter] = model->total_inertia;
        
        // Guardar historial de centroides
        for (int c = 0; c < model->num_clusters; c++) {
            for (int f = 0; f < model->num_features_trained; f++) {
                model->centroids_history[iter][c][f] = model->clusters[c].centroid[f];
            }
        }
        
        // Mostrar progreso
        if (iter % 10 == 0 || iter == max_iterations - 1) {
            printf("Iteración %3d: Inercia=%.4f, Puntos movidos=%d, Centroides movidos=%d\n",
                   iter + 1, model->total_inertia, points_moved, centroids_moved);
        }
        
        // Verificar convergencia
        if (centroids_moved == 0) {
            model->converged = 1;
            printf("\n✅ Convergencia alcanzada en la iteración %d\n", iter + 1);
            break;
        }
        
        // Verificar convergencia por umbral
        if (kmeans_has_converged(model, model->convergence_threshold)) {
            model->converged = 1;
            printf("\n✅ Convergencia (umbral) alcanzada en la iteración %d\n", iter + 1);
            break;
        }
    }
    
    clock_t end_time = clock();
    model->training_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    // Calcular métricas finales
    model->silhouette_score = calculate_silhouette_score(dataset, model);
    model->davies_bouldin_score = calculate_davies_bouldin_score(dataset, model);
    training_history.iteration_count = model->iterations;
    
    // Calcular radios de clusters
    for (int c = 0; c < model->num_clusters; c++) {
        double max_dist = 0.0;
        double sum_dist = 0.0;
        int count = 0;
        
        for (int i = 0; i < dataset->num_points; i++) {
            if (dataset->points[i].cluster_id == c) {
                double dist = euclidean_distance(dataset->points[i].features,
                                               model->clusters[c].centroid,
                                               model->num_features_trained);
                sum_dist += dist;
                if (dist > max_dist) max_dist = dist;
                count++;
            }
        }
        
        model->clusters[c].radius = (count > 0) ? sum_dist / count : 0.0;
        model->clusters[c].inertia = 0.0;
        
        // Calcular inercia del cluster
        for (int i = 0; i < dataset->num_points; i++) {
            if (dataset->points[i].cluster_id == c) {
                double dist = euclidean_distance(dataset->points[i].features,
                                               model->clusters[c].centroid,
                                               model->num_features_trained);
                model->clusters[c].inertia += dist * dist;
            }
        }
    }
    
    model->trained_at = time(NULL);
    
    // Generar nombre automático
    snprintf(model->name, sizeof(model->name), 
             "KMeans_K%d_%.0f", model->num_clusters, model->silhouette_score * 100);
    
    print_success("Entrenamiento completado!");
    printf("  • Tiempo: %.2f segundos\n", model->training_time);
    printf("  • Iteraciones: %d\n", model->iterations);
    printf("  • Inercia final: %.4f\n", model->total_inertia);
    printf("  • Puntuación de silueta: %.4f\n", model->silhouette_score);
    
    if (!model->converged && model->iterations == max_iterations) {
        print_warning("No se alcanzó convergencia en el máximo de iteraciones");
    }
}

void train_kmeans_step_by_step(Dataset* dataset, KMeans_Model* model) {
    print_section("ENTRENAMIENTO PASO A PASO DE K-MEANS");
    
    printf("Este modo muestra cada iteración del algoritmo K-Means.\n");
    printf("Podrás ver cómo los puntos se asignan a clusters y los centroides se mueven.\n\n");
    
    // Inicializar
    model->num_features_trained = dataset->num_features;
    model->convergence_threshold = 0.001;
    model->converged = 0;
    
    printf("PASO 1: INICIALIZACIÓN DE CENTROIDES\n");
    printf("Seleccionando %d puntos iniciales como centroides...\n", model->num_clusters);
    
    if (strcmp(model->initialization_method, "k-means++") == 0) {
        initialize_centroids_kmeansplusplus(dataset, model);
        printf("Método: k-means++ (mejor que aleatorio)\n");
    } else {
        initialize_centroids_random(dataset, model);
        printf("Método: aleatorio\n");
    }
    
    printf("\nCentroides iniciales seleccionados.\n");
    wait_for_enter();
    
    // Iteraciones
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        clear_screen();
        printf("ITERACIÓN %d\n\n", iter + 1);
        
        // Mostrar estado actual
        print_clustering_visualization_2d(dataset, model, iter);
        
        printf("\nPresiona Enter para ejecutar esta iteración...");
        getchar();
        
        // Ejecutar iteración
        int points_moved = assign_points_to_clusters(dataset, model);
        int centroids_moved = update_centroids(dataset, model);
        
        // Calcular inercia
        model->total_inertia = calculate_inertia(dataset, model);
        
        printf("\nResultados de la iteración %d:\n", iter + 1);
        printf("  • Puntos reasignados: %d\n", points_moved);
        printf("  • Centroides movidos: %d\n", centroids_moved);
        printf("  • Inercia actual: %.4f\n", model->total_inertia);
        
        // Verificar convergencia
        if (centroids_moved == 0) {
            printf("\n✅ ¡CONVERGENCIA ALCANZADA!\n");
            printf("Los centroides dejaron de moverse.\n");
            model->converged = 1;
            model->iterations = iter + 1;
            break;
        }
        
        if (iter == MAX_ITERATIONS - 1) {
            printf("\n⚠️  Máximo de iteraciones alcanzado\n");
            model->iterations = MAX_ITERATIONS;
        }
        
        printf("\nPresiona Enter para continuar a la siguiente iteración...");
        getchar();
    }
    
    // Calcular métricas finales
    model->silhouette_score = calculate_silhouette_score(dataset, model);
    model->davies_bouldin_score = calculate_davies_bouldin_score(dataset, model);
    model->trained_at = time(NULL);
    
    printf("\n🏁 ENTRENAMIENTO COMPLETADO\n");
    printf("  • Iteraciones totales: %d\n", model->iterations);
    printf("  • Inercia final: %.4f\n", model->total_inertia);
    printf("  • Silueta: %.4f\n", model->silhouette_score);
    
    wait_for_enter();
}

void train_kmeans_with_animation(Dataset* dataset, KMeans_Model* model) {
    print_section("ENTRENAMIENTO CON ANIMACIÓN");
    
    printf("Ejecutando K-Means con animación en tiempo real...\n\n");
    
    // Inicializar
    model->num_features_trained = dataset->num_features;
    model->convergence_threshold = 0.0001;
    
    if (strcmp(model->initialization_method, "k-means++") == 0) {
        initialize_centroids_kmeansplusplus(dataset, model);
    } else {
        initialize_centroids_random(dataset, model);
    }
    
    printf("Iniciando animación...\n");
    printf("(La animación se pausará brevemente entre iteraciones)\n\n");
    
    wait_for_key("Presiona Enter para comenzar...");
    
    // Bucle principal con animación
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        model->iterations = iter + 1;
        
        // Asignar puntos
        assign_points_to_clusters(dataset, model);
        
        // Mostrar estado actual
        print_clustering_visualization_2d(dataset, model, iter);
        
        // Pausa para animación
        usleep(animation_speed * 3);
        
        // Actualizar centroides
        int centroids_moved = update_centroids(dataset, model);
        
        // Calcular inercia
        model->total_inertia = calculate_inertia(dataset, model);
        
        // Verificar convergencia
        if (centroids_moved == 0) {
            model->converged = 1;
            
            // Mostrar estado final
            print_clustering_visualization_2d(dataset, model, iter);
            printf("\n✅ ¡CONVERGENCIA ALCANZADA!\n");
            break;
        }
        
        if (iter == MAX_ITERATIONS - 1) {
            printf("\n⚠️  Máximo de iteraciones alcanzado\n");
        }
    }
    
    // Calcular métricas finales
    model->silhouette_score = calculate_silhouette_score(dataset, model);
    model->davies_bouldin_score = calculate_davies_bouldin_score(dataset, model);
    model->trained_at = time(NULL);
    
    printf("\nAnimación completada.\n");
    printf("Iteraciones: %d, Inercia: %.4f, Silueta: %.4f\n",
           model->iterations, model->total_inertia, model->silhouette_score);
    
    wait_for_enter();
}

void find_optimal_k_elbow_method(Dataset* dataset, int k_min, int k_max) {
    print_section("BÚSQUEDA DE K ÓPTIMO (MÉTODO DEL CODO)");
    
    printf("Ejecutando K-Means para K desde %d hasta %d...\n", k_min, k_max);
    printf("Esto puede tomar unos momentos...\n\n");
    
    KMeans_Model models[MAX_CLUSTERS];
    int num_models = k_max - k_min + 1;
    
    for (int k = k_min; k <= k_max; k++) {
        printf("Probando K = %d... ", k);
        
        // Crear y entrenar modelo
        KMeans_Model model = {0};
        model.num_clusters = k;
        strcpy(model.initialization_method, "k-means++");
        
        train_kmeans(dataset, &model, 100);
        
        // Guardar modelo
        models[k - k_min] = model;
        
        printf("Inercia: %.2f, Silueta: %.3f\n", 
               model.total_inertia, model.silhouette_score);
    }
    
    // Mostrar gráfico de codo
    print_elbow_method_visualization(models, num_models);
    
    // Recomendar K óptimo
    double max_silhouette = -1.0;
    int best_k_by_silhouette = k_min;
    
    for (int i = 0; i < num_models; i++) {
        if (models[i].silhouette_score > max_silhouette) {
            max_silhouette = models[i].silhouette_score;
            best_k_by_silhouette = k_min + i;
        }
    }
    
    printf("\n🎯 RECOMENDACIÓN BASADA EN SILUETA:\n");
    printf("  • K óptimo: %d\n", best_k_by_silhouette);
    printf("  • Silueta: %.4f\n", max_silhouette);
    
    // Preguntar si usar este K
    printf("\n¿Deseas entrenar un modelo con K = %d? (s/n): ", best_k_by_silhouette);
    
    char respuesta;
    scanf("%c", &respuesta);
    getchar();
    
    if (respuesta == 's' || respuesta == 'S') {
        current_model = models[best_k_by_silhouette - k_min];
        print_success("Modelo con K óptimo cargado");
    }
    
    wait_for_enter();
}

// ============================ INTERFAZ ============================

void interactive_mode() {
    int choice;
    
    do {
        clear_screen();
        print_header("MODO INTERACTIVO PRINCIPAL - K-MEANS");
        
        printf("Selecciona una opción:\n\n");
        printf("1. 🎓 Modo Aprendizaje\n");
        printf("2. 🏋️  Entrenar Modelo K-Means\n");
        printf("3. 📊 Visualizar Clustering\n");
        printf("4. 📈 Análisis de Calidad\n");
        printf("5. 🔍 Encontrar K Óptimo\n");
        printf("6. 💾 Gestionar Modelos\n");
        printf("7. 📁 Gestionar Datasets\n");
        printf("8. 🧪 Modo Demo\n");
        printf("9. ⚙️  Configuración\n");
        printf("10. 🚪 Salir\n");
        
        printf("\nOpción: ");
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
                find_optimal_k_elbow_method(&current_dataset, 2, 10);
                break;
            case 6:
                model_management_menu();
                break;
            case 7:
                dataset_management_menu();
                break;
            case 8:
                demo_mode();
                break;
            case 9:
                settings_mode();
                break;
            case 10:
                printf("\nSaliendo...\n");
                break;
            default:
                print_error("Opción no válida");
                wait_for_enter();
        }
    } while (choice != 10);
}

void learning_mode_menu() {
    int choice;
    
    do {
        clear_screen();
        print_header("MODO APRENDIZAJE ACTIVO - K-MEANS");
        
        printf("Selecciona una actividad:\n\n");
        printf("1. 📚 Tutorial Interactivo\n");
        printf("2. 🧠 Conceptos Teóricos\n");
        printf("3. 👁️  Clustering Paso a Paso\n");
        printf("4. ❓ Cuestionario de Evaluación\n");
        printf("5. 🔍 Análisis de Casos Prácticos\n");
        printf("6. 🏠 Volver al Menú Principal\n");
        
        printf("\nOpción: ");
        scanf("%d", &choice);
        getchar();
        
        switch(choice) {
            case 1:
                interactive_tutorial();
                break;
            case 2:
                concept_explanation("kmeans_basics");
                break;
            case 3:
                step_by_step_clustering();
                break;
            case 4:
                take_quiz();
                break;
            case 5:
                explain_clustering_concepts(&current_model, &current_dataset);
                break;
            case 6:
                return;
            default:
                print_error("Opción no válida");
                wait_for_enter();
        }
    } while (choice != 6);
}

void training_mode() {
    clear_screen();
    print_header("ENTRENAMIENTO DE MODELO K-MEANS");
    
    if (current_dataset.num_points == 0) {
        print_error("No hay dataset cargado");
        wait_for_enter();
        return;
    }
    
    printf("Configurar parámetros del modelo:\n\n");
    
    // Pedir número de clusters
    printf("Número de clusters (K): ");
    int k;
    scanf("%d", &k);
    getchar();
    
    if (k < 1 || k > MAX_CLUSTERS) {
        print_error("K debe estar entre 1 y %d", MAX_CLUSTERS);
        wait_for_enter();
        return;
    }
    
    // Pedir método de inicialización
    printf("\nMétodo de inicialización:\n");
    printf("1. Aleatorio (simple)\n");
    printf("2. k-means++ (recomendado)\n");
    printf("Opción: ");
    
    int init_method;
    scanf("%d", &init_method);
    getchar();
    
    // Pedir modo de entrenamiento
    printf("\nModo de entrenamiento:\n");
    printf("1. Normal (rápido)\n");
    printf("2. Paso a paso (educativo)\n");
    printf("3. Con animación\n");
    printf("Opción: ");
    
    int training_mode_opt;
    scanf("%d", &training_mode_opt);
    getchar();
    
    // Configurar modelo
    current_model.num_clusters = k;
    
    if (init_method == 2) {
        strcpy(current_model.initialization_method, "k-means++");
    } else {
        strcpy(current_model.initialization_method, "random");
    }
    
    // Ejecutar entrenamiento según modo seleccionado
    switch(training_mode_opt) {
        case 1:
            train_kmeans(&current_dataset, &current_model, 100);
            break;
        case 2:
            train_kmeans_step_by_step(&current_dataset, &current_model);
            break;
        case 3:
            train_kmeans_with_animation(&current_dataset, &current_model);
            break;
        default:
            print_error("Opción no válida");
            wait_for_enter();
            return;
    }
    
    // Mostrar resultados
    print_model_info(&current_model);
    
    wait_for_enter();
}

void visualization_mode() {
    clear_screen();
    print_header("VISUALIZACIÓN DEL CLUSTERING");
    
    if (current_model.num_clusters == 0) {
        print_error("No hay modelo entrenado");
        wait_for_enter();
        return;
    }
    
    printf("Selecciona tipo de visualización:\n\n");
    printf("1. 🎨 Clustering Actual (2D)\n");
    printf("2. 📏 Información de Clusters\n");
    printf("3. 🚀 Animación de Movimiento de Centroides\n");
    printf("4. 📈 Análisis de Silueta por Punto\n");
    printf("5. 🗺️  Límites de Decisión\n");
    printf("6. 🔍 Análisis del Espacio de Características\n");
    printf("7. 🏠 Volver\n");
    
    printf("\nOpción: ");
    int choice;
    scanf("%d", &choice);
    getchar();
    
    switch(choice) {
        case 1:
            print_clustering_visualization_2d(&current_dataset, &current_model, 
                                            current_model.iterations - 1);
            break;
        case 2:
            print_clusters_info(&current_model);
            break;
        case 3:
            print_centroids_movement_animation(&current_model);
            break;
        case 4:
            print_silhouette_visualization(&current_dataset, &current_model);
            break;
        case 5:
            print_cluster_boundaries(&current_dataset, &current_model);
            break;
        case 6:
            print_feature_space_analysis(&current_dataset, &current_model);
            break;
        case 7:
            return;
        default:
            print_error("Opción no válida");
    }
    
    wait_for_enter();
}

void demo_mode() {
    clear_screen();
    print_header("MODO DEMOSTRACIÓN AUTOMÁTICA");
    
    printf("Este modo mostrará una demostración completa de K-Means.\n");
    printf("¿Comenzar demostración? (s/n): ");
    
    char respuesta;
    scanf("%c", &respuesta);
    getchar();
    
    if (respuesta != 's' && respuesta != 'S') {
        return;
    }
    
    // Paso 1: Crear dataset de demostración
    print_section("PASO 1: CREANDO DATASET DE DEMOSTRACIÓN");
    printf("Generando dataset con 4 clusters claramente separados...\n");
    
    current_dataset = create_random_clusters_dataset(200, 4, 0.8);
    normalize_dataset(&current_dataset);
    print_dataset_visualization(&current_dataset);
    
    wait_for_enter();
    
    // Paso 2: Entrenar con K correcto
    print_section("PASO 2: ENTRENANDO CON K CORRECTO (K=4)");
    printf("Usando el número correcto de clusters...\n");
    
    current_model.num_clusters = 4;
    strcpy(current_model.initialization_method, "k-means++");
    train_kmeans_with_animation(&current_dataset, &current_model);
    
    printf("\nObservación: Con K correcto, los clusters se forman naturalmente.\n");
    printf("Inercia baja y silueta alta indican buen clustering.\n");
    
    wait_for_enter();
    
    // Paso 3: Entrenar con K incorrecto
    print_section("PASO 3: ENTRENANDO CON K INCORRECTO (K=2)");
    printf("Usando muy pocos clusters...\n");
    
    KMeans_Model bad_model = {0};
    bad_model.num_clusters = 2;
    strcpy(bad_model.initialization_method, "k-means++");
    train_kmeans(&current_dataset, &bad_model, 50);
    
    printf("\nObservación: Con K muy pequeño, clusters naturales se fusionan.\n");
    printf("Inercia alta y silueta baja indican mal clustering.\n");
    
    wait_for_enter();
    
    // Paso 4: Método del codo
    print_section("PASO 4: MÉTODO DEL CODO PARA SELECCIONAR K");
    printf("Mostrando cómo seleccionar K usando el método del codo...\n");
    
    find_optimal_k_elbow_method(&current_dataset, 1, 8);
    
    // Paso 5: Comparación final
    print_section("PASO 5: COMPARACIÓN FINAL");
    
    printf("\n🎓 Lecciones aprendidas:\n");
    printf("1. K-Means busca minimizar la inercia (suma de distancias al cuadrado)\n");
    printf("2. La inicialización importa (k-means++ es mejor que aleatorio)\n");
    printf("3. Elegir K correcto es crucial\n");
    printf("4. El método del codo ayuda a seleccionar K\n");
    printf("5. La silueta mide la calidad del clustering\n");
    
    printf("\n📊 Métricas de calidad importantes:\n");
    printf("  • Inercia: Menor es mejor (pero cuidado con overfitting)\n");
    printf("  • Silueta: -1 a 1, mayor es mejor\n");
    printf("  • Davies-Bouldin: Menor es mejor\n");
    
    wait_for_enter();
}

void tutorial_mode() {
    clear_screen();
    print_header("TUTORIAL COMPLETO DE K-MEANS");
    
    printf("Bienvenido al tutorial completo de K-Means Clustering.\n");
    printf("Este tutorial cubrirá todos los conceptos paso a paso.\n\n");
    
    printf("¿Comenzar tutorial? (s/n): ");
    char respuesta;
    scanf("%c", &respuesta);
    getchar();
    
    if (respuesta != 's' || respuesta != 'S') {
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
                printf("📚 CONCEPTO 1: ¿Qué es K-Means?\n\n");
                printf("K-Means es un algoritmo de clustering no supervisado.\n");
                printf("Agrupa datos similares en K clusters.\n\n");
                printf("Características clave:\n");
                printf("  • No supervisado: No necesita etiquetas\n");
                printf("  • Basado en centroides: Cada cluster tiene un centro\n");
                printf("  • Iterativo: Mejora los clusters paso a paso\n");
                printf("  • Sensible a K: Necesita especificar número de clusters\n");
                break;
                
            case 2:
                printf("🔄 CONCEPTO 2: Cómo funciona K-Means\n\n");
                printf("1. INICIALIZACIÓN: Selecciona K centroides aleatorios\n");
                printf("2. ASIGNACIÓN: Cada punto va al centroide más cercano\n");
                printf("3. ACTUALIZACIÓN: Recalcula centroides como promedios\n");
                printf("4. REPETICIÓN: Hasta que centroides no cambien\n\n");
                printf("Objetivo: Minimizar la inercia (suma de distancias²)\n");
                break;
                
            case 3:
                printf("🎯 CONCEPTO 3: Elegir K (número de clusters)\n\n");
                printf("Problema: K-Means necesita que especifiques K.\n");
                printf("Soluciones:\n");
                printf("  • Conocimiento del dominio\n");
                printf("  • Método del codo (gráfico de inercia)\n");
                printf("  • Silueta (calidad de clustering)\n");
                printf("  • Prueba y error\n");
                break;
                
            case 4:
                printf("⚡ CONCEPTO 4: Inicialización k-means++\n\n");
                printf("Problema: Inicialización aleatoria da resultados inconsistentes.\n");
                printf("Solución: k-means++\n");
                printf("  • Primer centroide aleatorio\n");
                printf("  • Siguientes: Probabilidad proporcional a distancia²\n");
                printf("  • Resulta en mejor y más consistente clustering\n");
                break;
                
            case 5:
                printf("📊 CONCEPTO 5: Evaluar calidad de clustering\n\n");
                printf("Métricas importantes:\n");
                printf("  • Inercia: Suma de distancias² (menor es mejor)\n");
                printf("  • Silueta: -1 a 1 (mayor es mejor)\n");
                printf("  • Davies-Bouldin: Ratio de dispersión (menor es mejor)\n");
                printf("  • Estabilidad: Consistencia entre ejecuciones\n");
                break;
                
            case 6:
                printf("🚀 CONCEPTO 6: Aplicaciones prácticas\n\n");
                printf("K-Means se usa en:\n");
                printf("  • Segmentación de clientes\n");
                printf("  • Compresión de imágenes\n");
                printf("  • Análisis de documentos\n");
                printf("  • Detección de anomalías\n");
                printf("  • Bioinformática\n\n");
                printf("Limitaciones:\n");
                printf("  • Asume clusters esféricos\n");
                printf("  • Sensible a outliers\n");
                printf("  • Necesita especificar K\n");
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
    print_header("ANÁLISIS DE CALIDAD DE CLUSTERING");
    
    if (current_model.num_clusters == 0) {
        print_error("No hay modelo entrenado para analizar");
        wait_for_enter();
        return;
    }
    
    printf("Selecciona tipo de análisis:\n\n");
    printf("1. 📊 Métricas de Calidad\n");
    printf("2. 🔍 Distribución de Clusters\n");
    printf("3. 📈 Curva de Convergencia\n");
    printf("4. ⚠️  Detección de Problemas\n");
    printf("5. 💡 Sugerencias de Mejora\n");
    printf("6. 🏠 Volver\n");
    
    printf("\nOpción: ");
    int choice;
    scanf("%d", &choice);
    getchar();
    
    switch(choice) {
        case 1:
            print_cluster_metrics(&current_metrics, &current_model);
            break;
        case 2:
            print_clusters_info(&current_model);
            break;
        case 3:
            // Mostrar gráfico de convergencia
            printf("Curva de convergencia (Inercia vs Iteraciones):\n\n");
            
            if (training_history.iteration_count > 1) {
                int graph_width = 60;
                int graph_height = 15;
                
                // Encontrar máximo y mínimo
                double max_inertia = 0;
                double min_inertia = DBL_MAX;
                
                for (int i = 0; i < training_history.iteration_count; i++) {
                    if (training_history.inertia_history[i] > max_inertia)
                        max_inertia = training_history.inertia_history[i];
                    if (training_history.inertia_history[i] < min_inertia)
                        min_inertia = training_history.inertia_history[i];
                }
                
                // Dibujar gráfico
                for (int h = graph_height; h >= 0; h--) {
                    double inertia_value = max_inertia - (max_inertia - min_inertia) * h / graph_height;
                    printf("%8.2f │", inertia_value);
                    
                    for (int i = 0; i < training_history.iteration_count && i < graph_width; i++) {
                        double normalized = (training_history.inertia_history[i] - min_inertia) / 
                                          (max_inertia - min_inertia);
                        int pos = (int)(normalized * graph_height);
                        
                        if (h == pos) printf("●");
                        else if (h == 0) printf("─");
                        else printf(" ");
                    }
                    printf("\n");
                }
                
                printf("         └");
                for (int i = 0; i < graph_width && i < training_history.iteration_count; i++) printf("─");
                printf("→ Iteraciones\n");
                
                printf("\nMejora total: %.1f%%\n",
                       (training_history.inertia_history[0] - current_model.total_inertia) /
                       training_history.inertia_history[0] * 100);
            } else {
                printf("No hay suficiente historial de entrenamiento\n");
            }
            break;
        case 4:
            printf("Detección de problemas comunes:\n\n");
            
            // Clusters vacíos
            int empty_clusters = 0;
            for (int i = 0; i < current_model.num_clusters; i++) {
                if (current_model.clusters[i].point_count == 0) {
                    empty_clusters++;
                    printf("  • Cluster %d está vacío\n", i + 1);
                }
            }
            
            // Clusters desbalanceados
            int min_points = INT_MAX, max_points = 0;
            for (int i = 0; i < current_model.num_clusters; i++) {
                if (current_model.clusters[i].point_count < min_points)
                    min_points = current_model.clusters[i].point_count;
                if (current_model.clusters[i].point_count > max_points)
                    max_points = current_model.clusters[i].point_count;
            }
            
            if (max_points > min_points * 10 && min_points > 0) {
                printf("  • Clusters muy desbalanceados\n");
                printf("    El más grande tiene %d veces más puntos\n", max_points / min_points);
            }
            
            // Silueta baja
            if (current_model.silhouette_score < 0.3) {
                printf("  • Silueta baja (%.3f)\n", current_model.silhouette_score);
                printf("    Posible mala separación de clusters\n");
            }
            
            // Inercia muy alta
            double avg_distance = sqrt(current_model.total_inertia / current_dataset.num_points);
            if (avg_distance > 0.5) {
                printf("  • Distancia promedio alta (%.3f)\n", avg_distance);
                printf("    Los puntos están lejos de sus centroides\n");
            }
            
            if (empty_clusters == 0 && current_model.silhouette_score > 0.5) {
                printf("\n✅ No se detectaron problemas mayores\n");
            }
            break;
        case 5:
            printf("Sugerencias para mejorar el clustering:\n\n");
            
            if (current_model.silhouette_score < 0.3) {
                printf("1. Prueba con diferente valor de K\n");
                printf("   K actual: %d, Silueta: %.3f\n", 
                       current_model.num_clusters, current_model.silhouette_score);
            }
            
            if (strcmp(current_model.initialization_method, "random") == 0) {
                printf("2. Usa k-means++ para inicialización\n");
                printf("   Da resultados más consistentes\n");
            }
            
            // Verificar si hay clusters vacíos
            for (int i = 0; i < current_model.num_clusters; i++) {
                if (current_model.clusters[i].point_count == 0) {
                    printf("3. Cluster %d está vacío\n", i + 1);
                    printf("   Considera reducir K\n");
                    break;
                }
            }
            
            // Sugerir normalización si no está normalizado
            if (!current_dataset.is_normalized) {
                printf("4. Normaliza los datos\n");
                printf("   K-Means es sensible a escalas diferentes\n");
            }
            
            printf("\n🎯 Acciones recomendadas:\n");
            printf("  1. Usar método del codo para seleccionar K\n");
            printf("  2. Ejecutar múltiples veces con diferentes semillas\n");
            printf("  3. Evaluar con múltiples métricas\n");
            printf("  4. Visualizar resultados\n");
            break;
        case 6:
            return;
        default:
            print_error("Opción no válida");
    }
    
    wait_for_enter();
}

void model_management_menu() {
    int choice;
    
    do {
        clear_screen();
        print_header("GESTIÓN DE MODELOS K-MEANS");
        
        printf("Selecciona una opción:\n\n");
        printf("1. 💾 Guardar Modelo Actual\n");
        printf("2. 📂 Cargar Modelo\n");
        printf("3. 🖨️  Exportar Reporte\n");
        printf("4. ℹ️  Información del Modelo\n");
        printf("5. 🏠 Volver\n");
        
        printf("\nOpción: ");
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
                export_full_report(&current_model, &current_dataset, "reporte_kmeans.txt");
                break;
            case 4:
                print_model_info(&current_model);
                wait_for_enter();
                break;
            case 5:
                return;
            default:
                print_error("Opción no válida");
                wait_for_enter();
        }
    } while (choice != 5);
}

void dataset_management_menu() {
    int choice;
    
    do {
        clear_screen();
        print_header("GESTIÓN DE DATASETS");
        
        printf("Selecciona una opción:\n\n");
        printf("1. 📊 Información del Dataset\n");
        printf("2. 🎨 Visualizar Dataset\n");
        printf("3. 🔄 Generar Dataset de Prueba\n");
        printf("4. 💾 Guardar Dataset\n");
        printf("5. 🏠 Volver\n");
        
        printf("\nOpción: ");
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
                printf("1. Clusters aleatorios\n");
                printf("2. Círculos concéntricos\n");
                printf("3. Medias lunas\n");
                printf("4. Espirales\n");
                printf("5. Cancelar\n");
                
                int ds_choice;
                scanf("%d", &ds_choice);
                getchar();
                
                switch(ds_choice) {
                    case 1:
                        current_dataset = create_random_clusters_dataset(200, 4, 1.0);
                        normalize_dataset(&current_dataset);
                        print_success("Dataset de clusters aleatorios generado");
                        break;
                    case 2:
                        current_dataset = create_circular_clusters_dataset(200, 3);
                        normalize_dataset(&current_dataset);
                        print_success("Dataset de círculos concéntricos generado");
                        break;
                    case 3:
                        current_dataset = create_moon_shaped_dataset(200, 2);
                        normalize_dataset(&current_dataset);
                        print_success("Dataset de medias lunas generado");
                        break;
                    case 4:
                        current_dataset = create_spiral_dataset(200, 2);
                        normalize_dataset(&current_dataset);
                        print_success("Dataset de espirales generado");
                        break;
                    case 5:
                        break;
                    default:
                        print_error("Opción no válida");
                }
                wait_for_enter();
                break;
            case 4:
                {
                    char filename[256];
                    printf("Nombre del archivo (ej: dataset.csv): ");
                    scanf("%255s", filename);
                    getchar();
                    save_dataset(&current_dataset, filename);
                    wait_for_enter();
                }
                break;
            case 5:
                return;
            default:
                print_error("Opción no válida");
                wait_for_enter();
        }
    } while (choice != 5);
}

void settings_mode() {
    clear_screen();
    print_header("CONFIGURACIÓN DEL SISTEMA");
    
    printf("Configuración actual:\n");
    printf("  • Velocidad de animación: %d ms/frame\n", animation_speed / 1000);
    printf("  • Modo aprendizaje: %s\n", 
           learning_mode == 0 ? "Normal" : 
           learning_mode == 1 ? "Explicaciones" : "Paso a paso");
    
    printf("\nOpciones:\n");
    printf("1. Ajustar velocidad de animación\n");
    printf("2. Cambiar modo aprendizaje\n");
    printf("3. Restablecer configuración\n");
    printf("4. Volver\n");
    
    printf("\nOpción: ");
    int choice;
    scanf("%d", &choice);
    getchar();
    
    switch(choice) {
        case 1:
            printf("\nVelocidad de animación:\n");
            printf("1. Muy rápida (50 ms)\n");
            printf("2. Rápida (100 ms) - por defecto\n");
            printf("3. Normal (200 ms)\n");
            printf("4. Lenta (500 ms)\n");
            printf("Opción: ");
            
            int speed_choice;
            scanf("%d", &speed_choice);
            getchar();
            
            switch(speed_choice) {
                case 1: animation_speed = 50000; break;
                case 2: animation_speed = 100000; break;
                case 3: animation_speed = 200000; break;
                case 4: animation_speed = 500000; break;
                default: print_error("Opción no válida");
            }
            break;
            
        case 2:
            printf("\nModo aprendizaje:\n");
            printf("1. Normal (sin explicaciones extras)\n");
            printf("2. Con explicaciones\n");
            printf("3. Paso a paso (completo)\n");
            printf("Opción: ");
            
            int learn_choice;
            scanf("%d", &learn_choice);
            getchar();
            
            if (learn_choice >= 1 && learn_choice <= 3) {
                learning_mode = learn_choice - 1;
                print_success("Modo aprendizaje actualizado");
            } else {
                print_error("Opción no válida");
            }
            break;
            
        case 3:
            animation_speed = 100000;
            learning_mode = 0;
            print_success("Configuración restablecida a valores por defecto");
            break;
            
        case 4:
            return;
            
        default:
            print_error("Opción no válida");
    }
    
    wait_for_enter();
}

// ============================ FUNCIONES RESTANTES ============================

void load_quiz_questions() {
    // Pregunta 1
    strcpy(quiz_questions[0].question, "¿Cuál es el objetivo principal de K-Means?");
    strcpy(quiz_questions[0].options[0], "Predecir valores continuos");
    strcpy(quiz_questions[0].options[1], "Clasificar datos en categorías conocidas");
    strcpy(quiz_questions[0].options[2], "Agrupar datos similares sin etiquetas");
    strcpy(quiz_questions[0].options[3], "Encontrar correlaciones entre variables");
    quiz_questions[0].correct_answer = 2;
    strcpy(quiz_questions[0].explanation, 
           "K-Means es un algoritmo de CLUSTERING NO SUPERVISADO. "
           "Agrupa datos similares en clusters sin necesidad de etiquetas previas.");
    
    // Pregunta 2
    strcpy(quiz_questions[1].question, "¿Qué representa un centroide en K-Means?");
    strcpy(quiz_questions[1].options[0], "El punto más lejano del cluster");
    strcpy(quiz_questions[1].options[1], "Un punto aleatorio del dataset");
    strcpy(quiz_questions[1].options[2], "El punto promedio de todos los puntos del cluster");
    strcpy(quiz_questions[1].options[3], "El primer punto asignado al cluster");
    quiz_questions[1].correct_answer = 2;
    strcpy(quiz_questions[1].explanation,
           "El centroide es el PUNTO PROMEDIO de todos los puntos en un cluster. "
           "Se calcula como la media de las coordenadas de todos los puntos del cluster.");
    
    // Pregunta 3
    strcpy(quiz_questions[2].question, "¿Qué mide la inercia en K-Means?");
    strcpy(quiz_questions[2].options[0], "La velocidad de convergencia");
    strcpy(quiz_questions[2].options[1], "El número de clusters vacíos");
    strcpy(quiz_questions[2].options[2], "La suma de distancias al cuadrado de puntos a sus centroides");
    strcpy(quiz_questions[2].options[3], "La similitud entre clusters");
    quiz_questions[2].correct_answer = 2;
    strcpy(quiz_questions[2].explanation,
           "La INERCIA es la suma de las distancias al cuadrado de cada punto a su centroide asignado. "
           "K-Means busca minimizar esta métrica.");
    
    // Pregunta 4
    strcpy(quiz_questions[3].question, "¿Por qué es importante la inicialización en K-Means?");
    strcpy(quiz_questions[3].options[0], "No es importante, cualquier inicialización funciona igual");
    strcpy(quiz_questions[3].options[1], "Afecta la velocidad de convergencia pero no el resultado final");
    strcpy(quiz_questions[3].options[2], "Puede llevar a diferentes resultados finales (mínimos locales)");
    strcpy(quiz_questions[3].options[3], "Solo importa para datasets muy grandes");
    quiz_questions[3].correct_answer = 2;
    strcpy(quiz_questions[3].explanation,
           "K-Means es sensible a la INICIALIZACIÓN porque puede quedar atrapado en mínimos locales. "
           "k-means++ ayuda a evitar esto.");
    
    total_questions = 4;
}

void interactive_tutorial() {
    clear_screen();
    print_header("TUTORIAL INTERACTIVO DE K-MEANS");
    
    printf("Este tutorial te guiará paso a paso en el clustering con K-Means.\n\n");
    
    printf("Vamos a crear un dataset simple y aplicar K-Means.\n");
    wait_for_enter();
    
    // Crear dataset
    printf("1. Creando dataset con 3 clusters...\n");
    current_dataset = create_random_clusters_dataset(90, 3, 0.8);
    normalize_dataset(&current_dataset);
    print_dataset_visualization(&current_dataset);
    wait_for_enter();
    
    printf("2. Inicializando K-Means con K=3...\n");
    current_model.num_clusters = 3;
    strcpy(current_model.initialization_method, "k-means++");
    
    printf("3. Mostrando inicialización...\n");
    initialize_centroids_kmeansplusplus(&current_dataset, &current_model);
    print_clustering_visualization_2d(&current_dataset, &current_model, 0);
    wait_for_enter();
    
    printf("4. Ejecutando primera iteración...\n");
    assign_points_to_clusters(&current_dataset, &current_model);
    print_clustering_visualization_2d(&current_dataset, &current_model, 1);
    wait_for_enter();
    
    printf("5. Actualizando centroides...\n");
    update_centroids(&current_dataset, &current_model);
    print_clustering_visualization_2d(&current_dataset, &current_model, 2);
    wait_for_enter();
    
    printf("6. Completando entrenamiento...\n");
    train_kmeans(&current_dataset, &current_model, 10);
    wait_for_enter();
    
    printf("7. Analizando resultados...\n");
    print_clusters_info(&current_model);
    wait_for_enter();
    
    printf("\n🎓 Tutorial completado!\n");
    printf("Has aprendido:\n");
    printf("  • Cómo funciona el algoritmo K-Means\n");
    printf("  • Cómo se inicializan los centroides\n");
    printf("  • Cómo se asignan puntos a clusters\n");
    printf("  • Cómo se actualizan los centroides\n");
    printf("  • Cómo se evalúa la calidad del clustering\n");
    
    wait_for_enter();
}

void step_by_step_clustering() {
    clear_screen();
    print_header("CLUSTERING PASO A PASO");
    
    if (current_dataset.num_points == 0) {
        print_error("No hay dataset cargado");
        wait_for_enter();
        return;
    }
    
    printf("Este modo ejecutará K-Means mostrando cada paso detalladamente.\n");
    printf("¿Comenzar? (s/n): ");
    
    char respuesta;
    scanf("%c", &respuesta);
    getchar();
    
    if (respuesta != 's' || respuesta != 'S') {
        return;
    }
    
    // Configurar modelo
    printf("\nNúmero de clusters (K): ");
    int k;
    scanf("%d", &k);
    getchar();
    
    current_model.num_clusters = k;
    strcpy(current_model.initialization_method, "k-means++");
    
    // Ejecutar paso a paso
    train_kmeans_step_by_step(&current_dataset, &current_model);
}

void concept_explanation(const char* concept) {
    clear_screen();
    
    if (strcmp(concept, "kmeans_basics") == 0) {
        print_header("CONCEPTOS BÁSICOS DE K-MEANS");
        
        printf("📚 TEORÍA FUNDAMENTAL:\n\n");
        
        printf("1. CLUSTERING NO SUPERVISADO:\n");
        printf("   • No necesita etiquetas de entrenamiento\n");
        printf("   • Descubre estructura en los datos\n");
        printf("   • Agrupa puntos similares\n\n");
        
        printf("2. ALGORITMO K-MEANS:\n");
        printf("   • Input: Dataset X, número de clusters K\n");
        printf("   • Output: K clusters y sus centroides\n");
        printf("   • Objetivo: Minimizar inercia\n\n");
        
        printf("3. PSEUDOCÓDIGO:\n");
        printf("   1. Inicializar K centroides\n");
        printf("   2. REPETIR hasta convergencia:\n");
        printf("      a. Asignar cada punto al centroide más cercano\n");
        printf("      b. Recalcular centroides como promedios\n");
        printf("   3. DEVOLVER clusters y centroides\n\n");
        
        printf("4. MÉTRICAS DE EVALUACIÓN:\n");
        printf("   • Inercia: Σ ||x - μ||² (minimizar)\n");
        printf("   • Silueta: -1 a 1 (maximizar)\n");
        printf("   • Davies-Bouldin: Ratio de dispersión (minimizar)\n");
        
    } else if (strcmp(concept, "elbow_method") == 0) {
        print_header("MÉTODO DEL CODO");
        
        printf("📈 CÓMO SELECCIONAR K:\n\n");
        
        printf("Problema: K-Means necesita que especifiques K.\n\n");
        
        printf("Solución: Método del codo\n");
        printf("  1. Ejecutar K-Means para diferentes valores de K\n");
        printf("  2. Calcular inercia para cada K\n");
        printf("  3. Graficar K vs Inercia\n");
        printf("  4. Buscar el 'codo' donde añadir más K ya no reduce mucho la inercia\n\n");
        
        printf("📊 INTERPRETACIÓN DEL GRÁFICO:\n");
        printf("  • K pequeño: Inercia alta (underfitting)\n");
        printf("  • K adecuado: 'Codo' en la curva\n");
        printf("  • K grande: Inercia baja pero riesgo de overfitting\n\n");
        
        printf("💡 CONSEJOS PRÁCTICOS:\n");
        printf("  • Usar k-means++ para inicialización\n");
        printf("  • Ejecutar múltiples veces con diferentes semillas\n");
        printf("  • Considerar también la silueta\n");
        printf("  • Usar conocimiento del dominio cuando sea posible\n");
    }
    
    wait_for_enter();
}

void take_quiz() {
    print_header("EVALUACIÓN DE CONOCIMIENTOS K-MEANS");
    
    printf("Responde las siguientes preguntas para evaluar tu comprensión.\n");
    printf("Cada pregunta vale 1 punto. ¡Buena suerte!\n\n");
    
    for (int i = 0; i < total_questions; i++) {
        ask_question(&quiz_questions[i]);
    }
    
    printf("\n📊 Resultados del quiz:\n");
    printf("  • Puntaje: %d/%d\n", quiz_score, total_questions);
    printf("  • Porcentaje: %.1f%%\n", (double)quiz_score / total_questions * 100);
    
    if ((double)quiz_score / total_questions >= 0.7) {
        print_success("¡Excelente! Dominas los conceptos básicos de K-Means.");
    } else if ((double)quiz_score / total_questions >= 0.5) {
        print_warning("Buen trabajo, pero podrías repasar algunos conceptos.");
    } else {
        print_error("Necesitas estudiar más los conceptos de K-Means.");
    }
    
    wait_for_enter();
}

void ask_question(QuizQuestion* question) {
    print_section("PREGUNTA DE COMPRENSIÓN");
    
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
    
    printf("\n💡 Explicación: %s\n", question->explanation);
    wait_for_enter();
}

void explain_clustering_concepts(KMeans_Model* model, Dataset* dataset) {
    print_header("ANÁLISIS DE CONCEPTOS DE CLUSTERING");
    
    printf("1. CONCEPTO: Inicialización de centroides\n");
    printf("   • Aleatorio: Simple pero inconsistente\n");
    printf("   • k-means++: Mejora resultados, evita mínimos locales\n");
    printf("   • Tu modelo usa: %s\n\n", model->initialization_method);
    
    printf("2. CONCEPTO: Convergencia del algoritmo\n");
    printf("   • Tu modelo %sconvergió\n", model->converged ? "" : "NO ");
    printf("   • Iteraciones necesarias: %d\n", model->iterations);
    printf("   • Umbral de convergencia: %.6f\n\n", model->convergence_threshold);
    
    printf("3. CONCEPTO: Calidad del clustering\n");
    printf("   • Inercia: %.4f (menor es mejor)\n", model->total_inertia);
    printf("   • Silueta: %.4f (ideal > 0.5)\n", model->silhouette_score);
    
    if (model->silhouette_score > 0.7) {
        printf("   • ✅ Excelente separación de clusters\n");
    } else if (model->silhouette_score > 0.5) {
        printf("   • ⚠️  Separación aceptable\n");
    } else {
        printf("   • ❌ Separación pobre - considera cambiar K\n");
    }
    printf("\n");
    
    printf("4. CONCEPTO: Elección de K\n");
    printf("   • K actual: %d\n", model->num_clusters);
    printf("   • Método del codo: Ayuda a seleccionar K óptimo\n");
    printf("   • Silueta: Otra forma de evaluar K\n\n");
    
    printf("5. CONCEPTO: Limitaciones de K-Means\n");
    printf("   • Asume clusters esféricos y de tamaño similar\n");
    printf("   • Sensible a outliers\n");
    printf("   • Necesita especificar K\n");
    printf("   • Resultados dependen de inicialización\n");
    
    wait_for_enter();
}

void print_cluster_metrics(ClusterMetrics* metrics, KMeans_Model* model) {
    print_section("MÉTRICAS DE CALIDAD DE CLUSTERING");
    
    printf("📊 MÉTRICAS CALCULADAS:\n\n");
    
    printf("1. INERCIA (Within-Cluster Sum of Squares):\n");
    printf("   • Valor: %.4f\n", metrics->inertia);
    printf("   • Interpretación: ");
    if (metrics->inertia < 10) printf("Muy buena cohesión\n");
    else if (metrics->inertia < 50) printf("Buena cohesión\n");
    else if (metrics->inertia < 100) printf("Cohesión aceptable\n");
    else printf("Cohesión pobre\n");
    printf("\n");
    
    printf("2. PUNTUACIÓN DE SILUETA:\n");
    printf("   • Valor: %.4f\n", metrics->silhouette_score);
    printf("   • Rango: -1 (malo) a 1 (excelente)\n");
    printf("   • Interpretación: ");
    if (metrics->silhouette_score > 0.7) printf("Estructura fuerte\n");
    else if (metrics->silhouette_score > 0.5) printf("Estructura razonable\n");
    else if (metrics->silhouette_score > 0.25) printf("Estructura débil\n");
    else if (metrics->silhouette_score >= 0) printf("Sin estructura clara\n");
    else printf("Posible mala asignación\n");
    printf("\n");
    
    printf("3. ÍNDICE DE DAVIES-BOULDIN:\n");
    printf("   • Valor: %.4f\n", metrics->davies_bouldin);
    printf("   • Interpretación: ");
    if (metrics->davies_bouldin < 0.5) printf("Excelente separación\n");
    else if (metrics->davies_bouldin < 1.0) printf("Buena separación\n");
    else if (metrics->davies_bouldin < 2.0) printf("Separación aceptable\n");
    else printf("Separación pobre\n");
    printf("   • (Menor es mejor)\n\n");
    
    printf("4. DISTRIBUCIÓN DE PUNTOS POR CLUSTER:\n");
    for (int i = 0; i < model->num_clusters; i++) {
        printf("   • Cluster %d: %d puntos (%.1f%%)\n", 
               i + 1, 
               metrics->cluster_sizes[i],
               (double)metrics->cluster_sizes[i] / current_dataset.num_points * 100);
    }
    
    // Verificar balance
    int min_size = INT_MAX, max_size = 0;
    for (int i = 0; i < model->num_clusters; i++) {
        if (metrics->cluster_sizes[i] < min_size) min_size = metrics->cluster_sizes[i];
        if (metrics->cluster_sizes[i] > max_size) max_size = metrics->cluster_sizes[i];
    }
    
    double balance_ratio = (double)min_size / max_size;
    printf("\n   • Balance: %.2f ", balance_ratio);
    if (balance_ratio > 0.7) printf("✅ Muy balanceado\n");
    else if (balance_ratio > 0.3) printf("⚠️  Moderadamente balanceado\n");
    else printf("❌ Desbalanceado\n");
    
    printf("\n🎯 EVALUACIÓN GENERAL:\n");
    
    int good_metrics = 0;
    if (metrics->silhouette_score > 0.5) good_metrics++;
    if (metrics->davies_bouldin < 1.0) good_metrics++;
    if (balance_ratio > 0.3) good_metrics++;
    
    if (good_metrics == 3) {
        printf("  ✅ Excelente calidad de clustering\n");
    } else if (good_metrics >= 2) {
        printf("  ⚠️  Calidad aceptable\n");
    } else if (good_metrics >= 1) {
        printf("  ⚠️  Calidad marginal\n");
    } else {
        printf("  ❌ Calidad pobre - considera revisar parámetros\n");
    }
    
    printf("\n💡 RECOMENDACIONES:\n");
    if (metrics->silhouette_score < 0.3) {
        printf("  • Prueba con diferente valor de K\n");
    }
    if (balance_ratio < 0.2) {
        printf("  • Clusters muy desbalanceados\n");
        printf("  • Considera usar pesos o algoritmo diferente\n");
    }
    if (metrics->davies_bouldin > 2.0) {
        printf("  • Clusters muy superpuestos\n");
        printf("  • Considera reducir K o usar algoritmo no esférico\n");
    }
}

// ============================ PERSISTENCIA ============================

int save_model(KMeans_Model* model, const char* filename) {
    if (!filename || strlen(filename) == 0) {
        print_error("Nombre de archivo inválido");
        return 0;
    }
    
    FILE* file = fopen(filename, "wb");
    if (!file) {
        print_error("No se pudo crear el archivo");
        return 0;
    }
    
    // Escribir encabezado mágico
    fprintf(file, "KMEANS_MODEL_V1.0\n");
    
    // Escribir información básica
    fprintf(file, "NAME:%s\n", model->name);
    fprintf(file, "CLUSTERS:%d\n", model->num_clusters);
    fprintf(file, "INIT_METHOD:%s\n", model->initialization_method);
    fprintf(file, "FEATURES:%d\n", model->num_features_trained);
    fprintf(file, "ITERATIONS:%d\n", model->iterations);
    fprintf(file, "INERTIA:%f\n", model->total_inertia);
    fprintf(file, "SILHOUETTE:%f\n", model->silhouette_score);
    fprintf(file, "CONVERGED:%d\n", model->converged);
    fprintf(file, "CONVERGENCE_THRESH:%f\n", model->convergence_threshold);
    
    // Escribir centroides
    for (int c = 0; c < model->num_clusters; c++) {
        fprintf(file, "CENTROID_%d:", c);
        for (int f = 0; f < model->num_features_trained; f++) {
            fprintf(file, "%f", model->clusters[c].centroid[f]);
            if (f < model->num_features_trained - 1) fprintf(file, ",");
        }
        fprintf(file, "\n");
        
        fprintf(file, "CLUSTER_INFO_%d:%d,%f,%f\n", 
                c, 
                model->clusters[c].point_count,
                model->clusters[c].radius,
                model->clusters[c].inertia);
    }
    
    fclose(file);
    
    print_success("Modelo guardado exitosamente");
    return 1;
}

int load_model(KMeans_Model* model, const char* filename) {
    if (!filename || strlen(filename) == 0) {
        print_error("Nombre de archivo inválido");
        return 0;
    }
    
    FILE* file = fopen(filename, "rb");
    if (!file) {
        print_error("No se pudo abrir el archivo");
        return 0;
    }
    
    char line[1024];
    
    // Leer encabezado mágico
    if (!fgets(line, sizeof(line), file) || strstr(line, "KMEANS_MODEL") == NULL) {
        fclose(file);
        print_error("Formato de archivo inválido");
        return 0;
    }
    
    // Inicializar modelo
    memset(model, 0, sizeof(KMeans_Model));
    
    // Leer parámetros
    while (fgets(line, sizeof(line), file)) {
        line[strcspn(line, "\n")] = 0;  // Eliminar newline
        
        if (strncmp(line, "NAME:", 5) == 0) {
            strcpy(model->name, line + 5);
        } else if (strncmp(line, "CLUSTERS:", 9) == 0) {
            model->num_clusters = atoi(line + 9);
        } else if (strncmp(line, "INIT_METHOD:", 12) == 0) {
            strcpy(model->initialization_method, line + 12);
        } else if (strncmp(line, "FEATURES:", 9) == 0) {
            model->num_features_trained = atoi(line + 9);
        } else if (strncmp(line, "ITERATIONS:", 11) == 0) {
            model->iterations = atoi(line + 11);
        } else if (strncmp(line, "INERTIA:", 8) == 0) {
            model->total_inertia = atof(line + 8);
        } else if (strncmp(line, "SILHOUETTE:", 11) == 0) {
            model->silhouette_score = atof(line + 11);
        } else if (strncmp(line, "CONVERGED:", 10) == 0) {
            model->converged = atoi(line + 10);
        } else if (strncmp(line, "CONVERGENCE_THRESH:", 19) == 0) {
            model->convergence_threshold = atof(line + 19);
        } else if (strncmp(line, "CENTROID_", 9) == 0) {
            // Leer índice del cluster
            char* underscore = strchr(line + 9, ':');
            if (underscore) {
                int cluster_idx = atoi(line + 9);
                if (cluster_idx >= 0 && cluster_idx < model->num_clusters) {
                    char* values = underscore + 1;
                    char* token = strtok(values, ",");
                    int f = 0;
                    
                    while (token && f < model->num_features_trained) {
                        model->clusters[cluster_idx].centroid[f] = atof(token);
                        token = strtok(NULL, ",");
                        f++;
                    }
                }
            }
        } else if (strncmp(line, "CLUSTER_INFO_", 13) == 0) {
            // Leer información del cluster
            char* underscore = strchr(line + 13, ':');
            if (underscore) {
                int cluster_idx = atoi(line + 13);
                if (cluster_idx >= 0 && cluster_idx < model->num_clusters) {
                    char* values = underscore + 1;
                    char* token = strtok(values, ",");
                    
                    if (token) model->clusters[cluster_idx].point_count = atoi(token);
                    token = strtok(NULL, ",");
                    if (token) model->clusters[cluster_idx].radius = atof(token);
                    token = strtok(NULL, ",");
                    if (token) model->clusters[cluster_idx].inertia = atof(token);
                }
            }
        }
    }
    
    fclose(file);
    
    // Asignar colores y símbolos
    const char* colors[] = {COLOR_RED, COLOR_GREEN, COLOR_BLUE, COLOR_YELLOW, 
                           COLOR_MAGENTA, COLOR_CYAN, COLOR_WHITE};
    const char symbols[] = {'*', '#', '@', '+', 'x', 'o', 's'};
    
    for (int i = 0; i < model->num_clusters; i++) {
        strcpy(model->clusters[i].color_code, colors[i % 7]);
        model->clusters[i].symbol = symbols[i % 7];
        model->clusters[i].id = i;
    }
    
    return 1;
}

void save_model_interactive(KMeans_Model* model) {
    if (model->num_clusters == 0) {
        print_error("No hay modelo entrenado para guardar");
        wait_for_enter();
        return;
    }
    
    printf("Nombre del archivo para guardar (ej: modelo.km): ");
    char filename[256];
    scanf("%255s", filename);
    getchar();
    
    if (save_model(model, filename)) {
        strcpy(current_model_file, filename);
    }
    
    wait_for_enter();
}

void load_model_interactive(KMeans_Model* model) {
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

void export_full_report(KMeans_Model* model, Dataset* dataset, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        print_error("No se pudo crear el archivo de reporte");
        return;
    }
    
    fprintf(file, "========================================\n");
    fprintf(file, "        REPORTE COMPLETO K-MEANS\n");
    fprintf(file, "========================================\n\n");
    
    fprintf(file, "Fecha de generación: %s\n", ctime(&model->trained_at));
    
    // Información del modelo
    fprintf(file, "\n1. INFORMACIÓN DEL MODELO:\n");
    fprintf(file, "   • Nombre: %s\n", model->name);
    fprintf(file, "   • Método de inicialización: %s\n", model->initialization_method);
    fprintf(file, "   • Número de clusters (K): %d\n", model->num_clusters);
    fprintf(file, "   • Iteraciones: %d\n", model->iterations);
    fprintf(file, "   • Convergió: %s\n", model->converged ? "Sí" : "No");
    fprintf(file, "   • Inercia total: %.4f\n", model->total_inertia);
    fprintf(file, "   • Puntuación de silueta: %.4f\n", model->silhouette_score);
    
    // Información del dataset
    fprintf(file, "\n2. INFORMACIÓN DEL DATASET:\n");
    fprintf(file, "   • Puntos: %d\n", dataset->num_points);
    fprintf(file, "   • Características: %d\n", dataset->num_features);
    fprintf(file, "   • Normalizado: %s\n", dataset->is_normalized ? "Sí" : "No");
    
    // Información de cada cluster
    fprintf(file, "\n3. INFORMACIÓN POR CLUSTER:\n");
    fprintf(file, "┌─────┬──────────┬────────────┬────────────┬────────────┐\n");
    fprintf(file, "│  #  │  Puntos  │   Radio    │   Inercia  │  Porcentaje │\n");
    fprintf(file, "├─────┼──────────┼────────────┼────────────┼────────────┤\n");
    
    for (int i = 0; i < model->num_clusters; i++) {
        double percentage = (double)model->clusters[i].point_count / dataset->num_points * 100;
        fprintf(file, "│ %3d │ %8d │ %10.4f │ %10.4f │ %10.1f%% │\n",
                i + 1,
                model->clusters[i].point_count,
                model->clusters[i].radius,
                model->clusters[i].inertia,
                percentage);
    }
    fprintf(file, "└─────┴──────────┴────────────┴────────────┴────────────┘\n");
    
    // Centroides
    fprintf(file, "\n4. CENTROIDES FINALES:\n");
    for (int i = 0; i < model->num_clusters; i++) {
        fprintf(file, "   Cluster %d: [", i + 1);
        for (int f = 0; f < model->num_features_trained && f < 5; f++) {
            fprintf(file, "%.4f", model->clusters[i].centroid[f]);
            if (f < model->num_features_trained - 1 && f < 4) fprintf(file, ", ");
        }
        if (model->num_features_trained > 5) fprintf(file, ", ...");
        fprintf(file, "]\n");
    }
    
    // Evaluación
    fprintf(file, "\n5. EVALUACIÓN DE CALIDAD:\n");
    
    if (model->silhouette_score > 0.7) {
        fprintf(file, "   • Silueta: EXCELENTE (%.4f)\n", model->silhouette_score);
    } else if (model->silhouette_score > 0.5) {
        fprintf(file, "   • Silueta: BUENA (%.4f)\n", model->silhouette_score);
    } else if (model->silhouette_score > 0.25) {
        fprintf(file, "   • Silueta: ACEPTABLE (%.4f)\n", model->silhouette_score);
    } else {
        fprintf(file, "   • Silueta: POBRE (%.4f)\n", model->silhouette_score);
    }
    
    // Verificar clusters vacíos
    int empty_clusters = 0;
    for (int i = 0; i < model->num_clusters; i++) {
        if (model->clusters[i].point_count == 0) empty_clusters++;
    }
    
    if (empty_clusters > 0) {
        fprintf(file, "   • ⚠️  %d clusters vacíos\n", empty_clusters);
    } else {
        fprintf(file, "   • ✅ Todos los clusters tienen puntos\n");
    }
    
    // Recomendaciones
    fprintf(file, "\n6. RECOMENDACIONES:\n");
    
    if (model->silhouette_score < 0.3) {
        fprintf(file, "   • Considera probar con diferente valor de K\n");
    }
    
    if (strcmp(model->initialization_method, "random") == 0) {
        fprintf(file, "   • Considera usar k-means++ para mejor inicialización\n");
    }
    
    if (!model->converged && model->iterations >= 100) {
        fprintf(file, "   • El modelo no convergió completamente\n");
        fprintf(file, "   • Considera aumentar el número máximo de iteraciones\n");
    }
    
    fprintf(file, "\n========================================\n");
    fprintf(file, "        FIN DEL REPORTE\n");
    fprintf(file, "========================================\n");
    
    fclose(file);
    
    print_success("Reporte generado exitosamente");
    printf("Archivo: %s\n", filename);
    
    wait_for_enter();
}

void print_help() {
    print_header("AYUDA DEL SISTEMA K-MEANS DIDÁCTICO");
    
    printf("\nUso: programa [opciones]\n\n");
    printf("Opciones:\n");
    printf("  -i            Modo interactivo (por defecto)\n");
    printf("  -d ARCHIVO    Cargar dataset desde archivo CSV\n");
    printf("  -m ARCHIVO    Cargar modelo entrenado\n");
    printf("  -demo         Ejecutar demostración automática\n");
    printf("  -t            Modo tutorial paso a paso\n");
    printf("  -learn        Modo aprendizaje activo\n");
    printf("  -fast         Animaciones rápidas\n");
    printf("  -slow         Animaciones lentas\n");
    printf("  -h, --help    Mostrar esta ayuda\n");
    
    printf("\nEjemplos:\n");
    printf("  programa -d datos.csv          # Cargar dataset y entrenar\n");
    printf("  programa -m modelo.km          # Cargar modelo existente\n");
    printf("  programa -learn                # Modo aprendizaje guiado\n");
    printf("  programa -t                    # Tutorial completo\n");
    printf("  programa -fast -demo           # Demostración rápida\n");
}
