#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#define LOGV(v) std::cout << #v << "[" << v.rows() << " : " << 1 << "]" << std::endl
#define LOGM(v) std::cout << #v << "[" << v.rows() << " : " << v.cols() << "]" << std::endl
#define LOG(v) std::cout << #v << "=" << std::endl \
                         << v << std::endl

Eigen::MatrixXd readMatrix()
{
    // Open the file
    std::ifstream file("matrix.txt");
    if (!file.is_open()) {
        std::cout << "Error opening file." << std::endl;
        return Eigen::MatrixXd();
    }
    // Read the matrix from the file
    // Read the matrix dimensions
    int rows, cols;
    file >> rows >> cols;
    // Create a Eigen::MatrixXd to store the data
    Eigen::MatrixXd M(rows, cols);
    // Read the matrix elements
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file >> M(i, j);
        }
    }
    // Close the file
    file.close();
    return M;
}

// Генератор случайных векторов по среднему и ковариации
Eigen::VectorXd generateRandomVector(const Eigen::VectorXd& mean,
    const Eigen::MatrixXd& covariance)
{
    // Create random number generator
    std::random_device rd;
    std::mt19937 generator(rd());
    std::normal_distribution<double> distribution(0.0, 1.0); // Gaussian distribution with mean 0 and standard deviation 1
    // Compute Cholesky decomposition of the covariance matrix
    Eigen::LLT<Eigen::MatrixXd> llt(covariance);
    Eigen::MatrixXd L = llt.matrixL();

    // Generate random vector from standard Gaussian distribution
    Eigen::VectorXd z(mean.size());
    for (int i = 0; i < mean.size(); ++i) {
        z(i) = distribution(generator);
    }
    // Transform random vector to match the desired mean and covariance
    Eigen::VectorXd random_vector = mean + L * z;
    return random_vector;
}

// Эллипс по среднему и ковариации
// x - среднее
// P - ковариация
// n - количество сигм (масштаб)
// NP - количество точек эллипса
// возвращает NP точек эллипса + 4 точки для отрисовки осей
std::tuple<Eigen::VectorXd, Eigen::VectorXd> cov2elli(const Eigen::VectorXd& x, const Eigen::MatrixXd& P, int n = 3, int NP = 16)
{
    // +4 точки осей
    Eigen::VectorXd X(NP + 4);
    Eigen::VectorXd Y(NP + 4);
    double alpha = 2 * M_PI / NP;
    Eigen::MatrixXd circle(2, NP);
    for (int i = 0; i < NP; ++i) {
        circle(0, i) = cos(alpha * i);
        circle(1, i) = sin(alpha * i);
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(P, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd R = svd.matrixU();
    Eigen::MatrixXd D = svd.singularValues().asDiagonal();
    Eigen::MatrixXd d = D.array().sqrt().matrix();
    Eigen::MatrixXd ellip = n * R * d * circle;
    for (int i = 0; i < ellip.cols(); ++i) {
        X(i) = x(0) + ellip(0, i);
        Y(i) = x(1) + ellip(1, i);
    }
    Eigen::MatrixXd axis(2, 4);
    axis << -1, 1, 0, 0,
        0, 0, -1, 1;
    axis = (n * R * d * axis).colwise() + x;

    X.segment(NP, 4) = axis.row(0);
    Y.segment(NP, 4) = axis.row(1);
    return std::make_tuple(X, Y);
}
// Преобразование координат точки из глобальных в СК робота
Eigen::VectorXd toFrame(const Eigen::VectorXd& F,
    const Eigen::VectorXd& p,
    Eigen::MatrixXd& PF_f = Eigen::MatrixXd(),
    Eigen::MatrixXd& PF_p = Eigen::MatrixXd())
{
    Eigen::VectorXd pf;
    Eigen::VectorXd t = F.head(2);
    double a = F(2);
    Eigen::Matrix2d R;
    R << cos(a), -sin(a), sin(a), cos(a);
    pf = R.transpose() * (p - t);
    if (!PF_f.isZero() && !PF_p.isZero()) {
        double px = p(0);
        double py = p(1);
        double x = t(0);
        double y = t(1);
        PF_f << -cos(a), -sin(a), cos(a) * (py - y) - sin(a) * (px - x),
            sin(a), -cos(a), -cos(a) * (px - x) - sin(a) * (py - y);
        PF_p = R.transpose();
    }
    return pf;
}
// Преобразование координат точки из СК робота в глобальную.
Eigen::VectorXd fromFrame(const Eigen::VectorXd& F,
    const Eigen::VectorXd& pf,
    Eigen::MatrixXd& PW_f = Eigen::MatrixXd(),
    Eigen::MatrixXd& PW_pf = Eigen::MatrixXd())
{
    Eigen::VectorXd pw;
    Eigen::VectorXd t = F.head(2);
    double a = F(2);
    Eigen::Matrix2d R;
    R << cos(a), -sin(a), sin(a), cos(a);
    pw = R * pf + t.replicate(1, pf.cols());
    if (!PW_f.isZero() && !PW_pf.isZero()) {
        double px = pf(0);
        double py = pf(1);
        PW_f << 1, 0, -py * cos(a) - px * sin(a),
            0, 1, px * cos(a) - py * sin(a);
        PW_pf = R;
    }
    return pw;
}

// Направление и расстояние в x и y.
//
// вход:  y = [расстояние ; направление]
// Выход: p = [p x ; p y]
// P_y: Якобиан относительно y. Матрица 2x2
Eigen::VectorXd invScan(const Eigen::VectorXd& y, Eigen::MatrixXd& P_y = Eigen::MatrixXd())
{
    Eigen::VectorXd p;
    double d = y(0);
    double a = y(1);
    double px = d * cos(a);
    double py = d * sin(a);
    p.resize(2);
    p << px, py;
    if (!P_y.isZero()) {
        P_y.resize(2, 2);
        P_y << cos(a), -d * sin(a),
            sin(a), d * cos(a);
    }
    return p;
}

// Вход:  p : точка в СК робота p = [p x ; p y]
// Выход: y : измерение y = [range ; bearing]
// Y p: Якобиан относительно p
Eigen::VectorXd scan(const Eigen::VectorXd& p, Eigen::MatrixXd& Y_p = Eigen::MatrixXd())
{
    Eigen::VectorXd y;
    double px = p(0);
    double py = p(1);
    double d = sqrt(px * px + py * py);
    double a = atan2(py, px);
    y.resize(2);
    y << d, a;
    if (!Y_p.isZero()) {
        Y_p.resize(2, 2);
        Y_p << px / sqrt(px * px + py * py), py / sqrt(px * px + py * py),
            -py / (px * px * (py * py / (px * px) + 1)), 1 / (px * (py * py / (px * px) + 1));
    }
    return y;
}

// Преобразует измерение [расстояние;угол]
// в координаты точки в глобальной СК.
// Вход:
// r : СК роьота r = [r_x ; r_y ; r_alpha]
// y : измерение y = [расстояние ; направление]
// Выход:
// p : точка в СК роьота
// P_r: Якобиан относительно r
// P_y: Якобиан относительно  y
Eigen::VectorXd invObserve(const Eigen::VectorXd& r, const Eigen::VectorXd& y, Eigen::MatrixXd& P_r = Eigen::MatrixXd(), Eigen::MatrixXd& P_y = Eigen::MatrixXd())
{
    Eigen::VectorXd p;
    if (P_y.isZero()) {
        Eigen::VectorXd pf;
        Eigen::MatrixXd P_y;
        pf = invScan(y, P_y);
        Eigen::MatrixXd PW_f;
        Eigen::MatrixXd PW_pf;
        p = fromFrame(r, pf, PW_f, PW_pf);
    } else {
        Eigen::VectorXd p_r(2);
        Eigen::MatrixXd PR_y(2, 3), P_pr(2, 2);
        p_r = invScan(y, PR_y);
        p = fromFrame(r, p_r, P_r, P_pr);
        P_y = P_pr * PR_y;
    }
    return p;
}
// Преобразует точку в глобальной СК в измерение [расстояние;направление] .
// Вход:
// r : СК робота r = [r_x ; r_y ; r_alpha]
// p : точка в глобальной СК: p = [p_x ; p_y]
// Выход:
// y: измерение [расстояние;направление]
// Y r: Jacobian wrt r
// Y p: Якобиан относительно p
Eigen::VectorXd observe(const Eigen::VectorXd& r, const Eigen::VectorXd& p, Eigen::MatrixXd& Y_r = Eigen::MatrixXd(), Eigen::MatrixXd& Y_p = Eigen::MatrixXd())
{
    Eigen::VectorXd y;
    if (Y_r.isZero() && Y_p.isZero()) {
        Eigen::VectorXd pw(2);
        Eigen::MatrixXd PR_r(2, 3), PR_p;
        pw = toFrame(r, p, PR_r, PR_p);
        y = scan(pw, Y_p);
    } else {
        Eigen::VectorXd pr(2);
        Eigen::MatrixXd PR_r(2, 3), PR_p(2, 2);
        pr = toFrame(r, p, PR_r, PR_p);
        y = scan(pr, Y_p);
        Y_r = Y_p * PR_r;
        Y_p = Y_p * PR_p;
    }
    return y;
}

// Движение с раздельными входами управления и прогноза.
// Вход:
// r: СК робота r = [x ; y ; alpha]
// u: сигнал управления u = [d_x ; d_alpha]
// n: возмущения, доьавляемые к сигналу управления.
// Выход:
// ro: навая СК робота
// RO_r: Якобиан d(ro) / d(r)
// RO_n: Якобиан d(ro) / d(n)
Eigen::VectorXd moveRobot(const Eigen::VectorXd& r, const Eigen::VectorXd& u, const Eigen::VectorXd& n, Eigen::MatrixXd& RO_r = Eigen::MatrixXd(), Eigen::MatrixXd& RO_n = Eigen::MatrixXd())
{
    Eigen::VectorXd ro(3);
    double a = r(2);
    double dx = u(0) + n(0);
    double da = u(1) + n(1);
    double ao = a + da;
    if (ao > M_PI)
        ao -= 2 * M_PI;
    if (ao < -M_PI)
        ao += 2 * M_PI;
    Eigen::VectorXd dp(2);
    dp << dx, 0;
    if (RO_r.isZero() && RO_n.isZero()) {
        Eigen::MatrixXd PW_f, PW_pf;
        Eigen::VectorXd to(2);
        to = fromFrame(r, dp, PW_f, PW_pf);
        ro << to, ao;
    } else {
        Eigen::VectorXd to(2);
        Eigen::MatrixXd TO_r(2, 3), TO_dt(2, 2);
        to = fromFrame(r, dp, TO_r, TO_dt);

        double AO_a = 1;
        double AO_da = 1;
        RO_r.resize(3, 3);
        RO_r << TO_r, Eigen::Vector3d(0, 0, AO_a).transpose();
        RO_n.resize(3, 2);
        // RO_n << TO_dt.col(0),Vector2d(0,0) << Vector2d(0, AO_da);
        RO_n << TO_dt(0, 0), 0,
            TO_dt(1, 0), 0,
            0, AO_da;
        ro.resize(3);
        ro << to, ao;
    }
    return ro;
}

// Генериркут координаты маркеров
Eigen::MatrixXd cloister(double xmin, double xmax, double ymin, double ymax, int n = 9)
{
    Eigen::MatrixXd points = readMatrix();
    return points;
};

int findFreeSpaceInMap(Eigen::VectorXi& mapspace)
{
    int l = -1;
    for (int j = 0; j < mapspace.rows(); ++j) {
        if (mapspace(j) == false) {
            l = j;
            break;
        }
    }
    return l;
}
// ------------------------------------
// Точка входа
// ------------------------------------
int plotWorld(Eigen::MatrixXd& W,
    Eigen::VectorXd& R,
    Eigen::VectorXi& landmarks,
    Eigen::VectorXi& visible,
    Eigen::VectorXd& x,
    Eigen::MatrixXd& P);
int main()
{
    // -----------------------
    // Массив маркеров
    // -----------------------
    Eigen::MatrixXd W = cloister(-4, 4, -4, 4, 7);
    // кол-во макреров
    int N = W.cols();
    Eigen::VectorXi visible(N);
    visible.setConstant(0);
    std::vector<int> landmarks_to_add(N);
    std::iota(std::begin(landmarks_to_add), std::end(landmarks_to_add), 0);
    /*
    // Создаем генератор случайных чисел
    std::random_device rd;
    std::mt19937 g(rd()); // Используем mersenne_twister_engine
    // Перемешиваем элементы
    std::shuffle(landmarks_to_add.begin(), landmarks_to_add.end(), g);
    */
    // x,y,alpa - координаты и ориентация робота
    Eigen::VectorXd R(3);
    R << 0, -2, 0;
    // Управляющий сигнал
    Eigen::VectorXd U(2);
    // Вперед и поворот - движемся по кругу.
    U << 0.1, 0.05;
    // Измерения (расстояние и угол до марнера)
    Eigen::MatrixXd Y(2, N);
    Y.setZero();

    Eigen::VectorXd rm;
    // Вектор состояний всех объектов смстемы
    // 3 ячейки робот (x,y,angle)
    // 2*N ячеек для маркеров xm1,ym1,...xmN,ymN
    Eigen::VectorXd x = Eigen::VectorXd::Zero(3 + N * 2);
    // Матрица ковариации всех объектов системы
    Eigen::MatrixXd P = Eigen::MatrixXd::Zero(x.size(), x.size());
    // среднеквадратическое отклонение шума сигнала управления
    Eigen::VectorXd q(2);
    // среднеквадратическое отклонение шума измерения
    Eigen::VectorXd s(2);
    // маленькие значения приводят к вычислмтельной неустойчивости.
    q << 0.01, 0.01;
    // маленькие значения приводят к вычислмтельной неустойчивости.
    s << 0.1, 2 * M_PI / 180;
    // Ковариация шема сигналов управления
    Eigen::MatrixXd Q = q.cwiseProduct(q).asDiagonal();
    // Ковариацмя щума измерения
    Eigen::MatrixXd S = s.cwiseProduct(s).asDiagonal();
    Eigen::VectorXi mapspace(x.size());
    mapspace.setConstant(0);
    // индексы известных маккеров в массиве x (начинаются с 3, 0,1,2 - состояние робота)
    Eigen::VectorXi landmarks(N);
    landmarks.setConstant(-1);
    // Состояние робота в начале списка
    // отметим ячейки вектоа состоянмя как занятые
    Eigen::Vector3i r(0, 1, 2);
    mapspace(r).setConstant(1);
    // вносим состояние роьота
    x.segment(0, 3) = R;
    // инициализируем матрицу ковариации роьота
    P.block(0, 0, 3, 3).setConstant(0);
    // -----------------------
    // Главный цмкл
    // -----------------------
    Eigen::MatrixXd E_rl(2, 5);
    Eigen::MatrixXd E(2, 2);
    Eigen::VectorXd Yi(2);
    Eigen::VectorXd z(2);

    int key = 0;
    while (key != 27) {
        // вектор шума управления
        R = moveRobot(R, U, Eigen::Vector2d(0, 0), Eigen::MatrixXd(), Eigen::MatrixXd());
        // Измеряем расстояния и направления до маркеров,
        // добавляется шум измерения
        for (int i = 0; i < N; ++i) {
            auto obs = observe(R, W.col(i));

            // Проверим видит ли робот метку
            if (fabs(obs(1)) > M_PI / 6.0) {
                // если не видит, берум из прошлого прогноза
                // шум измерения, в данном случае отсутствует.
                if (landmarks(i) != -1) {
                    obs = observe(R, x.segment(landmarks(i), 2));
                    Y.col(i) = obs;
                }
                // отмечаем нивидимыи
                visible(i) = 0;
            } else {
                Y.col(i) = obs + generateRandomVector(Eigen::Vector2d(0, 0), S);
                // отмечаем видимыи
                visible(i) = 1;
            }
        }
        // Гипотеза
        // Сигнал управления с добавленным возмущений
        Eigen::MatrixXd R_r(3, 3);
        Eigen::MatrixXd R_n(3, 2);
        x.segment(0, 3) = moveRobot(x.segment(0, 3), U, generateRandomVector(Eigen::Vector2d(0, 0), Q), R_r, R_n);
        // заполняем матрмцу ковариации
        for (int i = 0; i < N; ++i) {
            // не открытые метки содержат -1
            // открытые - номер ячейки вектора состояний,
            // в которой лежит первый элемент вектора координат маркера
            if (landmarks(i) >= 0) {
                int l = landmarks(i);
                P.block(0, l, 3, 2) = R_r * P.block(0, l, 3, 2);
                P.block(l, 0, 2, 3) = P.block(0, l, 3, 2).transpose().eval();
            }
        }
        P.block(0, 0, 3, 3) = R_r * P.block(0, 0, 3, 3) * R_r.transpose() + R_n * Q * R_n.transpose();

        // Коррекция известных маркеров
        for (int li = 0; li < N; ++li) {
            int current_landmark = landmarks(li);
            if (current_landmark != -1) {
                // expectation: Gaussian fe,Eg
                // landmark pointer
                Eigen::MatrixXd E_r(3, 2), E_l(2, 2);
                auto e = observe(x.segment(0, 3), x.segment(current_landmark, 2), E_r, E_l);
                E_rl << E_r, E_l;
                // this is h(x) in EKF
                // pointers to robot and lmk.E_rl = [ E_r, E_l ];
                // expectation Jacobian
                Eigen::VectorXi rl(5);
                rl << 0, 1, 2, current_landmark, current_landmark + 1;

                E = E_rl * P(rl, rl) * E_rl.transpose();
                // measurement of landmark li
                Yi = Y.col(li);
                // innovation: Gaussian fz,Zg
                z = Yi - e; // this is z = y - h(x) in EKF
                // we need values around zero for angles:
                if (z(1) > M_PI) {
                    z(1) = z(1) - 2 * M_PI;
                }

                if (z(1) < -M_PI) {
                    z(1) = z(1) + 2 * M_PI;
                }
                Eigen::MatrixXd Z(2, 2);
                Z = S + E;

                // Individual compatibility check at Mahalanobis distance of 3-sigma
                rm = Eigen::VectorXd(mapspace.sum());
                // дальше маркеры

                int ind = 0;
                for (int i = 0; i < mapspace.rows(); ++i) {
                    if (mapspace(i)) {
                        rm(ind++) = i;
                    }
                }

                // Kalman gain
                {
                    // индексы состояния роьота и маркура в векторе состояний x.
                    // (для удоьного выдергивания подматриц из P и x.
                    Eigen::VectorXi rl(5);
                    rl << 0, 1, 2, current_landmark, current_landmark + 1;

                    Eigen::MatrixXd K(rm.size(), 2);
                    K = P(rm, rl) * E_rl.transpose() * Z.inverse(); // this is K = P*H' * Z ^ -1 in EKF
                    // обновляем карту
                    x(rm) = x(rm) + K * z;
                    P(rm, rm) = P(rm, rm) - K * Z * K.transpose();
                }
            }
        } // if current_landmark!=-1

        // ----------------------------------
        // Открываем точки
        // Порядок добавления определяется
        //  вектором landmarks_to_add
        // Каждая итерация добавляет 1 метку,
        // пока не будут добавлены все.
        // ----------------------------------
        for (int ii = 0; ii < landmarks_to_add.size(); ++ii) {
            if (visible(landmarks_to_add[ii])) {
                int idx = landmarks_to_add[ii];
                landmarks_to_add.erase(landmarks_to_add.begin() + ii);
                --ii;
                // ищем свободную ячейку в векторе состояний x
                int l = -1;
                l = findFreeSpaceInMap(mapspace);
                if (l == -1)
                    continue;
                // отмечаем занятыии
                mapspace.segment(l, 2).setConstant(true);
                // заносим индекс, где лежит наш маркер в векторе состояний
                landmarks(idx) = l;
                Eigen::MatrixXd L_r(2, 3);
                Eigen::MatrixXd L_y(2, 2);
                x.segment(l, 2) = invObserve(x.segment(0, 3), Y.col(idx), L_r, L_y);
                Eigen::Vector2i ll(l, l + 1);
                P(ll, rm) = L_r * P(r, rm);
                P(rm, ll) = P(ll, rm).transpose().eval();
                P.block(l, l, 2, 2) = L_r * P.block(0, 0, 3, 3) * L_r.transpose() + L_y * S * L_y.transpose();
            }
        }
        // ----------------------------------
        // Отрмсовка
        // ----------------------------------
        key = plotWorld(W, R, landmarks, visible, x, P);
    }

    return 0;
}
// для рисования треугольника с 2 равными сторонами
// по углу и сентру. Маркер роьота.
// Function to calculate the coordinates of A, B, and C of the triangle
void computeTrianglePoints(double AB, double BC, double CD,
    Eigen::Vector2d COM,
    double rotationAngle,
    Eigen::Vector2d& A,
    Eigen::Vector2d& B,
    Eigen::Vector2d& C)
{

    // Calculate relative coordinates of B and C with respect to A
    double theta = acos((AB * AB + BC * BC - CD * CD) / (2 * AB * BC));
    double beta = atan2(sqrt(3) * BC * sin(theta), AB + BC * cos(theta));
    B << AB * cos(beta), AB * sin(beta);
    C << BC * cos(theta + beta), BC * sin(theta + beta);
    auto cm = (B + C) / 3.0;
    // Rotate B and C around A
    Eigen::Matrix2d rotation;
    rotation << cos(rotationAngle), -sin(rotationAngle),
        sin(rotationAngle), cos(rotationAngle);
    A = -cm;
    B = B + A;
    C = C + A;
    A = rotation * A + COM;
    B = rotation * B + COM;
    C = rotation * C + COM;
}
// -----------------------------------
// Вся отрисовка ниже
// -----------------------------------
#include "opencv2/opencv.hpp"

// Перевод координат из мировых в экранные
cv::Point toCanvas(const Eigen::VectorXd& p)
{
    return cv::Point((p(0)) * 100 + 500, 500 - (p(1)) * 100);
}

// Рисуем эллипс ковариации с осями.
void drawEllipse(cv::Mat& img, float x, float y, const Eigen::MatrixXd& cov, bool visible = true)
{
    Eigen::VectorXd c(2);
    c << x, y;
    auto [X, Y] = cov2elli(c, cov);
    cv::Scalar color = visible ? cv::Scalar(255, 255, 255) : cv::Scalar(64, 64, 64);
    int thickness = visible ? 3 : 1;
    // Эллипс
    for (int i = 0; i < X.rows() - 4; ++i) {
        int i1 = (i + 1) % (X.rows() - 4);
        Eigen::VectorXd p1_(2), p2_(2);
        p1_ << X(i), Y(i);
        p2_ << X(i1), Y(i1);
        cv::Point p1 = toCanvas(p1_);
        cv::Point p2 = toCanvas(p2_);
        cv::line(img, p1, p2, color, thickness);
    }
    // Оси
    Eigen::VectorXd pa1_(2), pa2_(2), pb1_(2), pb2_(2);
    pa1_ << X(X.rows() - 4), Y(X.rows() - 4);
    pa2_ << X(X.rows() - 3), Y(X.rows() - 3);
    pb1_ << X(X.rows() - 2), Y(X.rows() - 2);
    pb2_ << X(X.rows() - 1), Y(X.rows() - 1);

    cv::Point pa1 = toCanvas(pa1_);
    cv::Point pa2 = toCanvas(pa2_);
    cv::Point pb1 = toCanvas(pb1_);
    cv::Point pb2 = toCanvas(pb2_);

    cv::line(img, pa1, pa2, cv::Scalar(255, 255, 255), 1);
    cv::line(img, pb1, pb2, cv::Scalar(255, 255, 255), 1);
}

// Рисуем маркеры
void drawLandmarks(cv::Mat& img, const Eigen::MatrixXd& Lms)
{
    for (int i = 0; i < Lms.cols(); ++i) {
        Eigen::VectorXd v(2);
        v << Lms(0, i), Lms(1, i);
        cv::Point p = toCanvas(v);
        cv::circle(img, p, 3, cv::Scalar(255, 255, 255), -1);
    }
}

// Рисуем мир

// Define the codec and frame size
int fourcc = cv::VideoWriter::fourcc('X', '2', '6', '4');
cv::Size frameSize(1000, 1000);
// Define the output video file name
const std::string outputFileName = "output.mp4";
cv::VideoWriter videoWriter;

int plotWorld(Eigen::MatrixXd& W, Eigen::VectorXd& R, Eigen::VectorXi& landmarks, Eigen::VectorXi& visible, Eigen::VectorXd& x, Eigen::MatrixXd& P)
{
    cv::Mat canvas(1000, 1000, CV_8UC3);
    // Create a VideoWriter object
    if (!videoWriter.isOpened()) {
        videoWriter.open(outputFileName, fourcc, 25, frameSize);
    }
    if (!videoWriter.isOpened()) {
        std::cout << "Error: Unable to open the output video file for writing." << std::endl;
        return -1;
    }

    int key = 0;

    canvas.setTo(cv::Scalar(0, 0, 0));
    Eigen::MatrixXd c(2, 2);
    c << P(0, 0), P(0, 1), P(1, 0), P(1, 1);
    drawLandmarks(canvas, W);
    drawEllipse(canvas, x(0), x(1), c);

    Eigen::Vector2d A, B, C;
    computeTrianglePoints(1, 0.5, 1, Eigen::Vector2d(x(0), x(1)), x(2), A, B, C);
    cv::line(canvas, toCanvas(static_cast<Eigen::VectorXd>(A)), toCanvas(static_cast<Eigen::VectorXd>(B)), cv::Scalar::all(255));
    cv::line(canvas, toCanvas(static_cast<Eigen::VectorXd>(B)), toCanvas(static_cast<Eigen::VectorXd>(C)), cv::Scalar::all(255));
    cv::line(canvas, toCanvas(static_cast<Eigen::VectorXd>(C)), toCanvas(static_cast<Eigen::VectorXd>(A)), cv::Scalar::all(255));

    Eigen::VectorXd rp(2);
    rp << R(0), R(1);
    cv::line(canvas, toCanvas(rp) - cv::Point(4, 0), toCanvas(rp) + cv::Point(4, 0), cv::Scalar::all(255));
    cv::line(canvas, toCanvas(rp) - cv::Point(0, 4), toCanvas(rp) + cv::Point(0, 4), cv::Scalar::all(255));

    for (int i = 0; i < landmarks.rows(); ++i) {
        int l = landmarks(i);
        if (l != -1) {
            Eigen::MatrixXd c(2, 2);
            c << P(l, l), P(l, l + 1), P(l + 1, l), P(l + 1, l + 1);
            drawEllipse(canvas, x(l), x(l + 1), c, visible(i));
        }
    }
    cv::imshow("canvas", canvas);
    key = cv::waitKey(20);
    videoWriter.write(canvas);
    if (key == 27) {
        // Release the VideoWriter object
        videoWriter.release();
        std::cout << "Video writing completed successfully." << std::endl;
    }
    return key;
}
