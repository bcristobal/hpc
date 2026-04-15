#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>
#include <limits.h>
#include <time.h>
#include <cuda_runtime.h>

// Compilar: nvcc grafoDistanciasCUDA.cu -o grafoDistanciasCUDA
// Ejecutar: time ./grafoDistanciasCUDA -n 2500 -w 20 -d 5 -c 15

/* -------------------- CONSTANTES -------------------- */
#define DEFAULT_NODES 1000
#define DEFAULT_MAX_WEIGHT 10
#define DEFAULT_MIN_EDGES 1
#define DEFAULT_MAX_EDGES 10

#define MAX_NODES 10000
#define MIN_NODES 2
#define MIN_WEIGHT 1
#define MAX_WEIGHT 100000
#define MIN_EDGES_PER_NODE 1
#define MAX_EDGES_PER_NODE 100

/* -------------------- ESTRUCTURAS -------------------- */
// Arista del grafo
typedef struct {
    int to;
    int weight;
} Edge;

// Nodo del grafo con lista de adyacencia
typedef struct {
    Edge edges[MAX_EDGES_PER_NODE];
    int num_edges;
} Node;

// Grafo global
Node graph[MAX_NODES];

/* -------------------- FUNCIONES -------------------- */

// Añade una arista al grafo
void add_edge(int from, int to, int weight) {
    int n = graph[from].num_edges;
    graph[from].edges[n].to = to;
    graph[from].edges[n].weight = weight;
    graph[from].num_edges++;
}

// Genera un grafo aleatorio
void generarGrafo(int nodes, int max_weight, int min_edges, int max_edges) {
    srand(time(NULL));
    // No se pareleliza, el random da problemas
    for(int i = 0; i < nodes; i++) {
        graph[i].num_edges = 0;
        int edges = rand() % (max_edges - min_edges + 1) + min_edges;
        for(int j = 0; j < edges; j++) {
            int to = rand() % nodes;
            int weight = rand() % max_weight + 1;
            add_edge(i, to, weight);
        }
    }
}

/* -------------------- KERNEL CUDA -------------------- */

// Kernel: cada hilo calcula Dijkstra para un nodo
__global__ void calcularEstadisticasKernel(Node *graph, int nodes, int *dist_all, int *visited_all, int *global_min, int *global_max, long long *global_sum, long long *global_count) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;      // Calcula el ID global del hilo (qué nodo le toca procesar)
    if (i >= nodes) return;                             // Evita que hilos extra (si hay más que nodos) hagan trabajo inválido

    int *dist = &dist_all[i * nodes];                   // Cada hilo usa su propia fila en la matriz dist_all
    int *visited = &visited_all[i * nodes];             // Cada hilo usa su propia fila en visited_all

    //-------- Método Inicio dijkstra --------
    for(int k = 0; k < nodes; k++) {
        dist[k] = INT_MAX;
        visited[k] = 0;
    }

    dist[i] = 0;

    for(int k = 0; k < nodes; k++) {

        // -------- Método Inicio distanciaMin --------
        int min = INT_MAX;
        int index = -1;

        for(int t = 0; t < nodes; t++) {
            if(!visited[t] && dist[t] < min) {
                min = dist[t];
                index = t;
            }
        }
        // -------- Método Fin distanciaMin --------

        if(index == -1) break;

        visited[index] = 1;

        for(int j = 0; j < graph[index].num_edges; j++) {
            int v = graph[index].edges[j].to;
            int weight = graph[index].edges[j].weight;

            if(!visited[v] && dist[index] + weight < dist[v]) {
                dist[v] = dist[index] + weight;
            }
        }
    }
    // -------- Método Fin dijkstra --------

    // Estadísticas locales por hilo
    int local_min = INT_MAX;
    int local_max = 0;
    long long local_sum = 0;
    long long local_count = 0;

    for(int j = 0; j < nodes; j++) {

        if(i == j) continue;
        if(dist[j] == INT_MAX) continue;

        int val = dist[j];

        if(val < local_min) local_min = val;
        if(val > local_max) local_max = val;

        local_sum += val;
        local_count++;
    }

    // Reducción global en GPU usando atomics
    atomicMin(global_min, local_min);       // Actualiza el mínimo global comparándolo con el mínimo local de este hilo
    atomicMax(global_max, local_max);
    atomicAdd(global_sum, local_sum);
    atomicAdd(global_count, local_count);
}

/* ======================== MAIN ======================= */
int main(int argc, char *argv[]) {

    int nodes = DEFAULT_NODES;
    int max_weight = DEFAULT_MAX_WEIGHT;
    int min_edges = DEFAULT_MIN_EDGES;
    int max_edges = DEFAULT_MAX_EDGES;

    int c;
    opterr = 0;

    while ((c = getopt(argc, argv, "n:w:d:c:")) != -1) {
        switch (c) {
            case 'n': nodes = atoi(optarg); break;
            case 'w': max_weight = atoi(optarg); break;
            case 'd': min_edges = atoi(optarg); break;
            case 'c': max_edges = atoi(optarg); break;
            case '?':
                fprintf(stderr, "Error en los argumentos. Forma de ejecutar el programa: %s [-n nodes] [-w max_weight] [-d min_edges] [-c max_edges]\n", argv[0]);
                return -1;
        }
    }

    /* VALIDACIÓN DE LÍMITES */
    if(nodes < MIN_NODES || nodes > MAX_NODES) {
        fprintf(stderr, "Error: número de nodos (%d) fuera del rango [%d, %d]\n", nodes, MIN_NODES, MAX_NODES);
        return -1;
    }

    if(max_weight < MIN_WEIGHT || max_weight > MAX_WEIGHT) {
        fprintf(stderr, "Error: peso máximo (%d) fuera del rango [%d, %d]\n", max_weight, MIN_WEIGHT, MAX_WEIGHT);
        return -1;
    }

    if(min_edges < MIN_EDGES_PER_NODE || max_edges > MAX_EDGES_PER_NODE || min_edges > max_edges) {
        fprintf(stderr, "Error: rango de aristas por nodo inválido [%d, %d]\n", MIN_EDGES_PER_NODE, MAX_EDGES_PER_NODE);
        return -1;
    }

    printf("Parámetros de ejecución:\n");
    printf("Nodos: %d\n", nodes);
    printf("Peso máximo: %d\n", max_weight);
    printf("Aristas mínimas por nodo: %d\n", min_edges);
    printf("Aristas máximas por nodo: %d\n\n", max_edges);

    generarGrafo(nodes, max_weight, min_edges, max_edges);

    /* -------------------- MEMORIA GPU -------------------- */

    Node *d_graph;
    int *d_dist, *d_visited;

    int *d_min, *d_max;
    long long *d_sum, *d_count;

    cudaMalloc(&d_graph, MAX_NODES * sizeof(Node));         // Reserva memoria en GPU para el grafo completo (array de nodos)
    cudaMalloc(&d_dist, nodes * nodes * sizeof(int));       // Reserva matriz N×N en GPU para almacenar distancias por cada hilo
    cudaMalloc(&d_visited, nodes * nodes * sizeof(int));    // Reserva matriz N×N en GPU para marcar nodos visitados por cada hilo

    cudaMalloc(&d_min, sizeof(int));                        // Reserva memoria en GPU para el mínimo global de distancias
    cudaMalloc(&d_max, sizeof(int));
    cudaMalloc(&d_sum, sizeof(long long));
    cudaMalloc(&d_count, sizeof(long long));

    cudaMemcpy(d_graph, graph, MAX_NODES * sizeof(Node), cudaMemcpyHostToDevice);   // Copia el grafo desde la RAM (CPU) a la memoria de la GPU

    /* Inicializar variables globales */
    int h_min = INT_MAX;
    int h_max = 0;
    long long h_sum = 0;
    long long h_count = 0;

    cudaMemcpy(d_min, &h_min, sizeof(int), cudaMemcpyHostToDevice);             // Inicializa en GPU el mínimo global con INT_MAX
    cudaMemcpy(d_max, &h_max, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum, &h_sum, sizeof(long long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, &h_count, sizeof(long long), cudaMemcpyHostToDevice);

    /* -------------------- LANZAMIENTO -------------------- */

    int threads = 256;
    int blocks = (nodes + threads - 1) / threads;

    calcularEstadisticasKernel<<<blocks, threads>>>(
        d_graph, nodes,
        d_dist, d_visited,
        d_min, d_max,
        d_sum, d_count
    );

    cudaDeviceSynchronize();

    /* -------------------- COPIA RESULTADOS -------------------- */

    cudaMemcpy(&h_min, d_min, sizeof(int), cudaMemcpyDeviceToHost);             // Copia el valor mínimo global desde la GPU a la CPU
    cudaMemcpy(&h_max, d_max, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_sum, d_sum, sizeof(long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_count, d_count, sizeof(long long), cudaMemcpyDeviceToHost);

    printf("=== Estadísticas de las distancias MÍNIMAS entre nodos ===\n");
    printf("Mínimo: %d\n", h_min);
    printf("Máximo: %d\n", h_max);
    printf("Media: %.2f\n", (double)h_sum / h_count);

    /* -------------------- LIMPIEZA -------------------- */

    cudaFree(d_graph);
    cudaFree(d_dist);
    cudaFree(d_visited);
    cudaFree(d_min);
    cudaFree(d_max);
    cudaFree(d_sum);
    cudaFree(d_count);

    return 0;
}