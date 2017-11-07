/*#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>

#define MASTER 0
#define UP 0
#define RIGHT 1
#define DOWN 2
#define LEFT 3

typedef struct Ctx_t {
    int l;
    int a;
    int b;
    int n;
    int N;
    double p_l;
    double p_r;
    double p_u;
    double p_d;
} Ctx;

typedef struct Particle_t {
    int x;
    int y;
    int n;
    int process;
} Particle;

void swap(void** x, void** y){
    void* tmp = *x;
    *x = *y;
    *y = tmp;
}

int step(Ctx* ctx);
void writeResult(Ctx* ctx, Particle* finished, int fin_size, int rank, int size);
void randomWalk(Ctx* ctx, int rank, int size);

int main (int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 10) {
        exit(1);
    }
    Ctx ctx = {
            .l = atoi(argv[1]),
            .a = atoi(argv[2]),
            .b = atoi(argv[3]),
            .n = atoi(argv[4]),
            .N = atoi(argv[5]),
            .p_l = atof(argv[6]),
            .p_r = atof(argv[7]),
            .p_u = atof(argv[8]),
            .p_d = atof(argv[9]),
    };

    if (ctx.a * ctx.b != size) {
        exit(1);
    }

    omp_set_num_threads(2);

    double ts1, ts2;
    if (rank == MASTER) {
        ts1 = MPI_Wtime();
    }

    randomWalk(&ctx, rank, size);

    if (rank == MASTER) {
        ts2 = MPI_Wtime();
        FILE* file;
        file = fopen("stats.txt", "w");
        fprintf(file, "%fs %d %d %d %d %d %f %f %f %f\n", ts2 - ts1, ctx.l, ctx.a, ctx.b, ctx.n, ctx.N,
                ctx.p_l, ctx.p_r, ctx.p_u, ctx.p_d);
        fclose(file);
    }

    MPI_Finalize();
    return 0;
}


int step(Ctx* ctx) {
    double p = rand() / RAND_MAX;
    if (p <= ctx->p_l) {
        return LEFT;
    } else if (p <= ctx->p_u) {
        return UP;
    } else if (p <= ctx->p_r) {
        return RIGHT;
    } else {
        return DOWN;
    }
}

void randomWalk(Ctx* ctx, int rank, int size) {
    int count = ctx->N;
    int max_count = 2 * count;
    Particle* particles = (Particle*) calloc(max_count, sizeof(Particle));

    int left_rank = rank - 1, right_rank = rank + 1, up_rank = rank - ctx->a, down_rank = rank + ctx->a;
    if (rank % ctx->a == 0) {
        left_rank += ctx->a;
    }
    if (rank % ctx->a == ctx->a - 1) {
        right_rank -= ctx->a;
    }
    if (rank / ctx->a == 0) {
        up_rank += size;
    }
    if (rank / ctx->a == ctx->b - 1) {
        down_rank = rank % ctx->a;
    }

    int left_size = 0, right_size = 0, up_size = 0, down_size = 0;
    int left_max_count = count, right_max_count = count, up_max_count = count, down_max_count = count;

    Particle* to_left = (Particle*) calloc(left_max_count, sizeof(Particle));
    Particle* to_right = (Particle*) calloc(right_max_count, sizeof(Particle));
    Particle* to_up = (Particle*) calloc(up_max_count, sizeof(Particle));
    Particle* to_down = (Particle*) calloc(down_max_count, sizeof(Particle));


    int fin_size = 0;
    int fin_max_count = count;
    Particle* finished = (Particle*) calloc(fin_max_count, sizeof(Particle));

    omp_lock_t lock;
    int is_running = 1;
    omp_init_lock(&lock);
    omp_set_lock(&lock);

#pragma omp parallel sections default(shared)
    {
#pragma omp section
        {
            while(is_running) {
                omp_set_lock(&lock);
                for (int j = 0; j < 100; j++) {
                    int i = 0;
                    while(i < count) {
                        if (particles[i].n == 0) {
                            if (fin_size >= fin_max_count) {
                                fin_max_count *= 2;
                                finished = (Particle*) realloc(finished, fin_max_count * sizeof(Particle));
                            }
                            finished[fin_size] = particles[i];
                            fin_size++;
                            particles[i] = particles[count - 1];
                            count--;
                            continue;
                        }
                        particles[i].n--;

                        switch(step(ctx)) {
                            case LEFT:
                                particles[i].x--;
                                if (particles[i].x < 0) {
                                    particles[i].x = ctx->l + particles[i].x;

                                    if (left_size >= left_max_count) {
                                        left_max_count *= 2;
                                        to_left = (Particle* )realloc(to_left, left_max_count * sizeof(Particle));
                                    }
                                    to_left[left_size] = particles[i];
                                    left_size++;

                                    particles[i] = particles[count - 1];
                                    count--;

                                    i--;
                                }
                                break;

                            case RIGHT:
                                particles[i].x++;
                                if (particles[i].x >= ctx->l) {
                                    particles[i].x = ctx->l - particles[i].x;

                                    if (right_size >= right_max_count) {
                                        left_max_count *= 2;
                                        to_right = (Particle* )realloc(to_right, right_max_count * sizeof(Particle));
                                    }
                                    to_right[right_size] = particles[i];
                                    right_size++;

                                    particles[i] = particles[count - 1];
                                    count--;

                                    i--;
                                }
                                break;
                            case DOWN:
                                particles[i].y--;
                                if (particles[i].y < 0) {
                                    particles[i].y = ctx->l + particles[i].y;

                                    if (down_size >= down_max_count) {
                                        down_max_count *= 2;
                                        to_down = (Particle* )realloc(to_down, down_max_count * sizeof(Particle));
                                    }
                                    to_down[down_size] = particles[i];
                                    down_size++;

                                    particles[i] = particles[count - 1];
                                    count--;

                                    i--;
                                }
                                break;
                            case UP:
                                particles[i].y++;
                                if (particles[i].y >= ctx->l) {
                                    particles[i].y = ctx->l - particles[i].y;

                                    if (up_size >= up_max_count) {
                                        up_max_count *= 2;
                                        to_up = (Particle* )realloc(to_up, up_max_count * sizeof(Particle));
                                    }
                                    to_up[up_size] = particles[i];
                                    up_size++;

                                    particles[i] = particles[count - 1];
                                    count--;

                                    i--;
                                }
                                break;
                            default:
                                break;
                        }
                        i++;
                    }
                }
                omp_unset_lock(&lock);
            }
        }

#pragma omp section
        {
            int* seeds;
            int seed;
            if (rank == MASTER) {
                srand(time(NULL));
                seeds = (int*) calloc(size, sizeof(int));
                for (int i = 0; i < size; i++) {
                    seeds[i] = rand();
                }
            }
            MPI_Scatter(seeds, 1, MPI_INT, &seed, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
            if (rank == MASTER) {
                free(seeds);
            }
            srand(seed);

            for (int i = 0; i < count; i++) {
                int x = rand() % ctx->l;
                int y = rand() % ctx->l;
                Particle tmp_particle = {
                        .x = x,
                        .y = y,
                        .n = ctx->n,
                        .process = rank,
                };
                particles[i] = tmp_particle;
            }

            int tmp_left_size = left_size, tmp_right_size = right_size, tmp_up_size = up_size, tmp_down_size = down_size;
            int tmp_left_max_count = left_max_count, tmp_right_max_count = right_max_count,
                    tmp_up_max_count = up_max_count, tmp_down_max_count = down_max_count;

            Particle* tmp_left = (Particle*) calloc(tmp_left_max_count, sizeof(Particle));
            Particle* tmp_right = (Particle*) calloc(tmp_right_max_count, sizeof(Particle));
            Particle* tmp_up = (Particle*) calloc(tmp_up_max_count, sizeof(Particle));
            Particle* tmp_down = (Particle*) calloc(tmp_down_max_count, sizeof(Particle));
            int tmp_finished_size = fin_size;

            omp_unset_lock(&lock);

            while (is_running) {
                omp_set_lock(&lock);
                swap((void*) &tmp_left_size, (void*) &tmp_left_size);
                swap((void*) &tmp_right_size, (void*) &right_size);
                swap((void*) &tmp_up_size, (void*) &up_size);
                swap((void*) &tmp_down_size, (void*) &down_size);
                swap((void*) &tmp_left_max_count, (void*) &left_max_count);
                swap((void*) &tmp_right_max_count, (void*) &right_max_count);
                swap((void*) &tmp_up_max_count, (void*) &up_max_count);
                swap((void*) &tmp_down_max_count, (void*) &down_max_count);
                left_size = 0;
                right_size = 0;
                up_size = 0;
                down_size = 0;
                swap((void*) &tmp_left, (void*) &to_left);
                swap((void*) &tmp_right, (void*) &to_right);
                swap((void*) &tmp_up, (void*) &to_up);
                swap((void*) &tmp_down, (void*) &to_down);
                tmp_finished_size = fin_size;

                omp_unset_lock(&lock);

                MPI_Request requests[8];

                int from_left_size, from_right_size, from_up_size, from_down_size;
                MPI_Issend(&tmp_left_size, 1, MPI_INT, left_rank, LEFT, MPI_COMM_WORLD, &requests);
                MPI_Issend(&tmp_right_size, 1, MPI_INT, right_rank, RIGHT, MPI_COMM_WORLD, &requests + 1);
                MPI_Issend(&tmp_up_size, 1, MPI_INT, up_rank, UP, MPI_COMM_WORLD, &requests + 2);
                MPI_Issend(&tmp_down_size, 1, MPI_INT, down_rank, DOWN, MPI_COMM_WORLD, &requests + 3);

                MPI_Irecv(&from_left_size, 1, MPI_INT, left_rank, RIGHT, MPI_COMM_WORLD, &requests + 4);
                MPI_Irecv(&from_right_size, 1, MPI_INT, right_rank, LEFT, MPI_COMM_WORLD, &requests + 5);
                MPI_Irecv(&from_up_size, 1, MPI_INT, up_rank, DOWN, MPI_COMM_WORLD, &requests + 6);
                MPI_Irecv(&from_down_size, 1, MPI_INT, down_rank, UP, MPI_COMM_WORLD, &requests + 7);

                MPI_Waitall(8, &requests, MPI_STATUS_IGNORE);

                Particle* from_left = (Particle*) calloc(from_left_size, sizeof(Particle));
                Particle* from_right = (Particle*) calloc(from_right_size, sizeof(Particle));
                Particle* from_up = (Particle*) calloc(from_up_size, sizeof(Particle));
                Particle* from_down = (Particle*) calloc(from_down_size, sizeof(Particle));

                MPI_Issend(tmp_left, tmp_left_size * sizeof(Particle), MPI_BYTE,
                           left_rank, LEFT, MPI_COMM_WORLD, &requests);
                MPI_Issend(tmp_right, tmp_right_size * sizeof(Particle), MPI_BYTE,
                           right_rank, RIGHT, MPI_COMM_WORLD, &requests + 1);
                MPI_Issend(tmp_up, tmp_up_size * sizeof(Particle), MPI_BYTE,
                           up_rank, UP, MPI_COMM_WORLD, &requests + 2);
                MPI_Issend(tmp_down, tmp_down_size * sizeof(Particle), MPI_BYTE,
                           down_rank, DOWN, MPI_COMM_WORLD, &requests + 3);

                MPI_Irecv(from_left, from_left_size * sizeof(Particle), MPI_BYTE,
                          left_rank, RIGHT, MPI_COMM_WORLD, &requests + 4);
                MPI_Irecv(from_right, from_right_size * sizeof(Particle), MPI_BYTE,
                          right_rank, LEFT, MPI_COMM_WORLD, &requests + 5);
                MPI_Irecv(from_up, from_up_size * sizeof(Particle), MPI_BYTE,
                          up_rank, DOWN, MPI_COMM_WORLD, &requests + 6);
                MPI_Irecv(from_down, from_down_size * sizeof(Particle), MPI_BYTE,
                          down_rank, UP, MPI_COMM_WORLD, &requests + 7);

                MPI_Waitall(8, &requests, MPI_STATUS_IGNORE);

                int all_finished[size];
                MPI_Gather(&tmp_finished_size, 1, MPI_INT, all_finished, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

                omp_set_lock(&lock);

                for (int i = 0; i < from_left_size; i++) {
                    if (count >= max_count) {
                        max_count *= 2;
                        particles = (Particle* )realloc(particles, max_count * sizeof(Particle));
                    }
                    particles[count] = from_left[i];
                    count++;
                }

                for (int i = 0; i < from_right_size; i++) {
                    if (count >= max_count) {
                        max_count *= 2;
                        particles = (Particle* )realloc(particles, max_count * sizeof(Particle));
                    }
                    particles[count] = from_right[i];
                    count++;
                }

                for (int i = 0; i < from_up_size; i++) {
                    if (count >= max_count) {
                        max_count *= 2;
                        particles = (Particle* )realloc(particles, max_count * sizeof(Particle));
                    }
                    particles[count] = from_up[i];
                    count++;
                }

                for (int i = 0; i < from_down_size; i++) {
                    if (count >= max_count) {
                        max_count *= 2;
                        particles = (Particle* )realloc(particles, max_count * sizeof(Particle));
                    }
                    particles[count] = from_down[i];
                    count++;
                }

                int is_actives[size];

                if (rank == MASTER) {

                    int sum_finished = 0;
                    for (int i = 0; i < size; i++) {
                        sum_finished += all_finished[i];
                    }

                    if (sum_finished == size * ctx->N) {
                        for (int i = 0; i < size; i++) {
                            is_actives[i] = 0;
                        }
                    } else {
                        for (int i = 0; i < size; i++) {
                            is_actives[i] = 1;
                        }
                    }
                }

                MPI_Scatter(is_actives, 1, MPI_INT, &is_running, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

                omp_unset_lock(&lock);

                free(from_left);
                free(from_right);
                free(from_up);
                free(from_down);
            }

            writeResult(ctx, finished, fin_size, rank, size);

            free(tmp_left);
            free(tmp_right);
            free(tmp_up);
            free(tmp_down);
        }
    }

    free(particles);
    free(to_left);
    free(to_right);
    free(to_up);
    free(to_down);
    free(finished);
    omp_destroy_lock(&lock);
}

void writeResult(Ctx* ctx, Particle* finished, int fin_size, int rank, int size) {
    MPI_File data;
    MPI_File_delete("data.bin", MPI_INFO_NULL);
    MPI_File_open(MPI_COMM_WORLD, "data.bin", MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &data);

    int positions[ctx->l][ctx->l * size];
    for (int y = 0; y < ctx->l; y++) {
        for (int x = 0; x < ctx->l * size; x++) {
            positions[y][x] = 0;
        }
    }

    for (int i = 0; i < fin_size; i++) {
        positions[finished[i].y][finished[i].x * size + finished[i].process] += 1;
    }

    int start_seek = ((ctx->l * ctx->l) * rank / ctx->a * ctx->a + ctx->l * (rank % ctx->a)) * sizeof (int) * size;
    int line_seek = (ctx->l * ctx->a) * sizeof(int) * size;

    for (int y = 0; y < ctx->l; y++) {
        MPI_File_set_view(data, start_seek + line_seek * y, MPI_INT, MPI_INT, "native", MPI_INFO_NULL);
        MPI_File_write(data, positions[y], ctx->l * size, MPI_INT, MPI_STATUS_IGNORE);;
    }

    MPI_File_close(&data);
}
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>

#define MASTER 0
#define UP 0
#define RIGHT 1
#define DOWN 2
#define LEFT 3

typedef struct ctx_t {
    int l;
    int a;
    int b;
    int n;
    int N;
    double pl;
    double pr;
    double pu;
    double pd;
} ctx_t;

typedef struct particle_t {
    int x;
    int y;
    int n;
    int process;
} particle_t;

void swap_particle_array(particle_t** x, particle_t** y) {
    particle_t* tmp = *x;
    *x = *y;
    *y = tmp;
}

void swap_int(int* x, int* y) {
    int tmp = *x;
    *x = *y;
    *y = tmp;
}


int get_dir(double left, double up, double right, double down) {
    if (left >= up && left >= right && left >= down) {
        return LEFT;
    } else if (up >= left && up >= right && up >= down) {
        return UP;
    } else if (right >= left && right >= up && right >= down) {
        return RIGHT;
    } else {
        return DOWN;
    }
}

int step(ctx_t* ctx) {
    double left = rand() * ctx->pl;
    double right = rand() * ctx->pr;
    double up = rand() * ctx->pu;
    double down = rand() * ctx->pd;

    return get_dir(left, up, right, down);
}

void insert(particle_t x, particle_t** ar, int* size, int* max_size) {
    if (*size >= *max_size) {
        *max_size *= 2;
        *ar = (particle_t* )realloc(*ar, (*max_size) * sizeof(particle_t));
    }
    (*ar)[*size] = x;
    (*size)++;
}

void delete(int index, particle_t** ar, int* size) {
    (*ar)[index] = (*ar)[(*size)-1];
    (*size)--;
}

void write_data_to_file(ctx_t* ctx, particle_t* finished, int size, int rank, int comm_size) {
    MPI_File data;
    MPI_File_delete("data.txt", MPI_INFO_NULL);
    MPI_File_open(MPI_COMM_WORLD, "data.txt", MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &data);

    int a = rank % ctx->a;
    int b = rank / ctx->a;

    int positions[ctx->l][ctx->l * comm_size];
    for (int y = 0; y < ctx->l; y++) {
        for (int x = 0; x < ctx->l * comm_size; x++) {
            positions[y][x] = 0;
        }
    }

    for (int i = 0; i < size; i++) {
        positions[finished[i].y][finished[i].x * comm_size + finished[i].process] += 1;
    }

    int start_seek = ((ctx->l * ctx->l) * b * ctx->a + ctx->l * a) * sizeof (int) * comm_size;
    int line_seek = (ctx->l * ctx->a) * sizeof(int) * comm_size;

    for (int y = 0; y < ctx->l; y++) {
        MPI_File_set_view(data, start_seek + line_seek * y, MPI_INT, MPI_INT, "native", MPI_INFO_NULL);
        MPI_File_write(data, positions[y], ctx->l * comm_size, MPI_INT, MPI_STATUS_IGNORE);;
    }

    MPI_File_close(&data);
}

void process(ctx_t* ctx, int rank, int comm_size) {
    int a = rank % ctx->a;
    int b = rank / ctx->a;
    int left_rank, right_rank, up_rank, down_rank;
    if (rank % ctx->a != 0) {
        left_rank = rank - 1;
    } else {
        left_rank = rank + ctx->a - 1;
    }

    if (rank % ctx->a != ctx->a - 1) {
        right_rank = rank + 1;
    } else {
        right_rank = rank - (ctx->a - 1);
    }

    if (rank / ctx->a != 0) {
        up_rank = rank - ctx->a;
    } else {
        up_rank = comm_size - ctx->a + rank;
    }

    if (rank / ctx->a != ctx->b-1) {
        down_rank = rank + ctx->a;
    } else {
        down_rank = rank % ctx->a;
    }

    int size = ctx->N;
    int max_size = 2 * size;
    particle_t* particles = (particle_t*) malloc(max_size * sizeof(particle_t));

    int left_size = 0;
    int right_size = 0;
    int up_size = 0;
    int down_size = 0;
    int left_max_size = size;
    int right_max_size = size;
    int up_max_size = size;
    int down_max_size = size;
    particle_t* to_left = (particle_t*) malloc(left_max_size * sizeof(particle_t));
    particle_t* to_right = (particle_t*) malloc(right_max_size * sizeof(particle_t));
    particle_t* to_up = (particle_t*) malloc(up_max_size * sizeof(particle_t));
    particle_t* to_down = (particle_t*) malloc(down_max_size * sizeof(particle_t));

    int fin_size = 0;
    int fin_max_size = size;
    particle_t* finished = (particle_t*) malloc(fin_max_size * sizeof(particle_t));

    omp_lock_t lock;
    int is_running = 1;
    omp_init_lock(&lock);
    omp_set_lock(&lock);

#pragma omp parallel sections default(shared)
    {
#pragma omp section
        {
            while(is_running) {
                omp_set_lock(&lock);
                for (int j = 0; j < 100; j++) {
                    int i = 0;
                    while(i < size) {
                        if (particles[i].n == 0) {
                            insert(particles[i], &finished, &fin_size, &fin_max_size);
                            delete(i, &particles, &size);
                            continue;
                        }
                        particles[i].n--;
                        int dir = step(ctx);
                        switch(dir) {
                            case LEFT:
                                particles[i].x--;
                                if (particles[i].x < 0) {
                                    particles[i].x = ctx->l + particles[i].x;
                                    insert(particles[i], &to_left, &left_size, &left_max_size);
                                    delete(i, &particles, &size);
                                    i--;
                                }
                                break;
                            case RIGHT:
                                particles[i].x++;
                                if (particles[i].x >= ctx->l) {
                                    particles[i].x = ctx->l - particles[i].x;
                                    insert(particles[i], &to_right, &right_size, &right_max_size);
                                    delete(i, &particles, &size);
                                    i--;
                                }
                                break;
                            case UP:
                                particles[i].y--;
                                if (particles[i].y < 0) {
                                    particles[i].y = ctx->l + particles[i].y;
                                    insert(particles[i], &to_up, &up_size, &up_max_size);
                                    delete(i, &particles, &size);
                                    i--;
                                }
                                break;
                            case DOWN:
                                particles[i].y++;
                                if (particles[i].y >= ctx->l) {
                                    particles[i].y = ctx->l - particles[i].y;
                                    insert(particles[i], &to_down, &down_size, &down_max_size);
                                    delete(i, &particles, &size);
                                    i--;
                                }
                                break;
                            default:
                                break;
                        }
                        i++;
                    }
                }
                omp_unset_lock(&lock);
            }
        }

#pragma omp section
        {
            int* seeds;
            int seed;
            if (rank == MASTER) {
                srand(time(NULL));
                seeds = (int*) malloc (comm_size * sizeof(int));
                for (int i = 0; i < comm_size; i++) {
                    seeds[i] = rand();
                }
            }
            MPI_Scatter(seeds, 1, MPI_INT, &seed, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
            if (rank == MASTER) {
                free(seeds);
            }
            srand(seed);

            for (int i = 0; i < size; i++) {
                int x = rand() % ctx->l;
                int y = rand() % ctx->l;
                particle_t tmp_particle = {
                        .x = x,
                        .y = y,
                        .n = ctx->n,
                        .process = rank,
                };
                particles[i] = tmp_particle;
            }

            int tmp_left_size = left_size;
            int tmp_right_size = right_size;
            int tmp_up_size = up_size;
            int tmp_down_size = down_size;
            int tmp_left_max_size = left_max_size;
            int tmp_right_max_size = right_max_size;
            int tmp_up_max_size = up_max_size;
            int tmp_down_max_size = down_max_size;

            particle_t* tmp_left = malloc(tmp_left_max_size * sizeof(particle_t));
            particle_t* tmp_right = malloc(tmp_right_max_size * sizeof(particle_t));
            particle_t* tmp_up = malloc(tmp_up_max_size * sizeof(particle_t));
            particle_t* tmp_down = malloc(tmp_down_max_size * sizeof(particle_t));
            int tmp_finished_size = fin_size;

            omp_unset_lock(&lock);

            while (is_running) {
                omp_set_lock(&lock);
                swap_int(&tmp_left_size, &left_size);
                swap_int(&tmp_right_size, &right_size);
                swap_int(&tmp_up_size, &up_size);
                swap_int(&tmp_down_size, &down_size);
                swap_int(&tmp_left_max_size, &left_max_size);
                swap_int(&tmp_right_max_size, &right_max_size);
                swap_int(&tmp_up_max_size, &up_max_size);
                swap_int(&tmp_down_max_size, &down_max_size);
                left_size = 0;
                right_size = 0;
                up_size = 0;
                down_size = 0;
                swap_particle_array(&tmp_left, &to_left);
                swap_particle_array(&tmp_right, &to_right);
                swap_particle_array(&tmp_up, &to_up);
                swap_particle_array(&tmp_down, &to_down);
                tmp_finished_size = fin_size;

                omp_unset_lock(&lock);

                MPI_Request requests[8];

                int from_left_size, from_right_size, from_up_size, from_down_size;
                MPI_Issend(&tmp_left_size, 1, MPI_INT, left_rank, LEFT, MPI_COMM_WORLD, requests);
                MPI_Issend(&tmp_right_size, 1, MPI_INT, right_rank, RIGHT, MPI_COMM_WORLD, requests + 1);
                MPI_Issend(&tmp_up_size, 1, MPI_INT, up_rank, UP, MPI_COMM_WORLD, requests + 2);
                MPI_Issend(&tmp_down_size, 1, MPI_INT, down_rank, DOWN, MPI_COMM_WORLD, requests + 3);

                MPI_Irecv(&from_left_size, 1, MPI_INT, left_rank, RIGHT, MPI_COMM_WORLD, requests + 4);
                MPI_Irecv(&from_right_size, 1, MPI_INT, right_rank, LEFT, MPI_COMM_WORLD, requests + 5);
                MPI_Irecv(&from_up_size, 1, MPI_INT, up_rank, DOWN, MPI_COMM_WORLD, requests + 6);
                MPI_Irecv(&from_down_size, 1, MPI_INT, down_rank, UP, MPI_COMM_WORLD, requests + 7);

                MPI_Waitall(8, requests, MPI_STATUS_IGNORE);

                particle_t* from_left = (particle_t*) malloc(from_left_size * sizeof(particle_t));
                particle_t* from_right = (particle_t*) malloc(from_right_size * sizeof(particle_t));
                particle_t* from_up = (particle_t*) malloc(from_up_size * sizeof(particle_t));
                particle_t* from_down = (particle_t*) malloc(from_down_size * sizeof(particle_t));

                MPI_Issend(tmp_left, tmp_left_size * sizeof(particle_t), MPI_BYTE, left_rank, LEFT, MPI_COMM_WORLD, requests);
                MPI_Issend(tmp_right, tmp_right_size * sizeof(particle_t), MPI_BYTE, right_rank, RIGHT, MPI_COMM_WORLD, requests + 1);
                MPI_Issend(tmp_up, tmp_up_size * sizeof(particle_t), MPI_BYTE, up_rank, UP, MPI_COMM_WORLD, requests + 2);
                MPI_Issend(tmp_down, tmp_down_size * sizeof(particle_t), MPI_BYTE, down_rank, DOWN, MPI_COMM_WORLD, requests + 3);

                MPI_Irecv(from_left, from_left_size * sizeof(particle_t), MPI_BYTE, left_rank, RIGHT, MPI_COMM_WORLD, requests + 4);
                MPI_Irecv(from_right, from_right_size * sizeof(particle_t), MPI_BYTE, right_rank, LEFT, MPI_COMM_WORLD, requests + 5);
                MPI_Irecv(from_up, from_up_size * sizeof(particle_t), MPI_BYTE, up_rank, DOWN, MPI_COMM_WORLD, requests + 6);
                MPI_Irecv(from_down, from_down_size * sizeof(particle_t), MPI_BYTE, down_rank, UP, MPI_COMM_WORLD, requests + 7);

                MPI_Waitall(8, requests, MPI_STATUS_IGNORE);

                int all_finished[comm_size];
                MPI_Gather(&tmp_finished_size, 1, MPI_INT, all_finished, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

                omp_set_lock(&lock);

                for (int i = 0; i < from_left_size; i++) {
                    insert(from_left[i], &particles, &size, &max_size);
                }

                for (int i = 0; i < from_right_size; i++) {
                    insert(from_right[i], &particles, &size, &max_size);
                }

                for (int i = 0; i < from_up_size; i++) {
                    insert(from_up[i], &particles, &size, &max_size);
                }

                for (int i = 0; i < from_down_size; i++) {
                    insert(from_down[i], &particles, &size, &max_size);
                }

                int is_actives[comm_size];

                if (rank == MASTER) {

                    int sum_finished = 0;
                    for (int i = 0; i < comm_size; i++) {
                        sum_finished += all_finished[i];
                    }

                    if (sum_finished == comm_size * ctx->N) {
                        for (int i = 0; i < comm_size; i++) {
                            is_actives[i] = 0;
                        }
                    } else {
                        for (int i = 0; i < comm_size; i++) {
                            is_actives[i] = 1;
                        }
                    }
                }

                MPI_Scatter(is_actives, 1, MPI_INT, &is_running, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

                omp_unset_lock(&lock);

                free(from_left);
                free(from_right);
                free(from_up);
                free(from_down);
            }

            write_data_to_file(ctx, finished, fin_size, rank, comm_size);

            free(tmp_left);
            free(tmp_right);
            free(tmp_up);
            free(tmp_down);
        }
    }

    free(particles);
    free(to_left);
    free(to_right);
    free(to_up);
    free(to_down);
    free(finished);
    omp_destroy_lock(&lock);
}

int main (int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 10) {
        if (rank == MASTER) {
            printf("Incorrect number of arguments\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    ctx_t ctx = {
            .l = atoi(argv[1]),
            .a = atoi(argv[2]),
            .b = atoi(argv[3]),
            .n = atoi(argv[4]),
            .N = atoi(argv[5]),
            .pl = atof(argv[6]),
            .pr = atof(argv[7]),
            .pu = atof(argv[8]),
            .pd = atof(argv[9]),
    };

    if (ctx.a * ctx.b != size) {
        if (rank == MASTER) {
            printf("Incorrect number of processes\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    omp_set_num_threads(2);

    double start_time, end_time;
    if (rank == MASTER) {
        start_time = MPI_Wtime();
    }
    process(&ctx, rank, size);
    if (rank == MASTER) {
        end_time = MPI_Wtime();
        FILE* stats = fopen("stats.txt", "w");

        fprintf(stats, "%fs %d %d %d %d %d %f %f %f %f\n", end_time - start_time,
                ctx.l, ctx.a, ctx.b, ctx.n, ctx.N, ctx.pl, ctx.pr, ctx.pu, ctx.pd);

        fclose(stats);
    }

    MPI_Finalize();
    return 0;
}