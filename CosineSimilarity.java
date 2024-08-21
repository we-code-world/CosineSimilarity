package test;

import java.util.*;

public class CosineSimilarity extends Throwable{
    // 表示多维向量的数据结构
    public static class tensor{
        private int[] shape;
        private double[] data;
        private tensor cache;
        public tensor(){
            shape = null;
            data = null;
        }
        public int getShapeNum(int[] shape){
            int num = 1;
            for (int j : shape){
                if(j <= 0) throw new IllegalArgumentException("shape中的值必须为正数");
                num *= j;
            }
            return num;
        }
        public tensor(int[] shape){
            this.shape = shape;
            int num = getShapeNum(shape);
            data = new double[num];
        }
        public tensor(int[] shape, double[] data){
            this.shape = shape;
            int num = getShapeNum(shape);
            assert num == data.length: "初始化数据出错";
            this.data = data;
        }
        public int getDimension(int dim){
            if (dim < 0) dim = shape.length + dim;
            if (dim >= data.length) throw new IndexOutOfBoundsException("维度超出范围");
            return shape[dim];
        }

        public List<tensor> getTensors(int dim){
            if (dim < 0) dim = shape.length + dim;
            int length = getDimension(dim);
            List<tensor> tensors = new ArrayList<>(length + 1);
            int left = 1, right = 1;
            int[] shape_i = Arrays.copyOf(shape, shape.length - 1);
            for (int i = 0; i < shape.length; i++){
                if(i < dim) left *= shape[i];
                else if(i > dim) right *= shape[i];
            }
            for (int i = dim + 1; i < shape.length; i++) shape_i[i - 1] = shape[i];
            for (int i = 0; i < length; i ++) {
                double[] data_i = new double[right * left];
                for (int j = 0; j < data_i.length; j++) data_i[j] = data[(j / right * length + i) * right + j % right];
                tensors.add(new tensor(shape_i, data_i));
            }
            return tensors;
        }
        // 计算输出形状，参考torch的实现从后向前匹配
        private int[] calculate_out_shape(tensor b){
            int[] rtn_shape;
            if (shape.length >= b.shape.length) rtn_shape = Arrays.copyOf(shape, shape.length);
            else rtn_shape = Arrays.copyOf(b.shape, b.shape.length);
            int min_length = Math.min(shape.length, b.shape.length);
            int idx_a = shape.length;
            int idx_b = b.shape.length;
            for (int i = 1; i <= min_length; i++) {
                idx_a --;
                idx_b --;
                if(shape[idx_a] != b.shape[idx_b]){
                    if (shape[idx_a] == 1) rtn_shape[rtn_shape.length - i] = b.shape[idx_b];
                    else if (b.shape[idx_b] == 1) rtn_shape[rtn_shape.length - i] = shape[idx_a];
                    else throw new IndexOutOfBoundsException(String.format("tensor a 的第%d维度与tensor b 的第%d维不匹配" , idx_a, idx_b));
                }
            }
            return rtn_shape;
        }
        private void set_cache(tensor b){
            cache = b;
        }
        private void empty_cache(){
            cache = null;
        }
        private void traverse_item(int level, int start, int cache_start, char op){
            if(level == shape.length){
                switch (op){
                    case '+':
                        data[start] += cache.data[cache_start];
                        break;
                    case '*':
                        data[start] *= cache.data[cache_start];
                        break;
                    case '/':
                        if(cache.data[cache_start] == 0) data[start] = 0.;
                        else data[start] /= cache.data[cache_start];
                        break;
                    default:
                        data[start] = cache.data[cache_start];
                }
            }
            else{
                for (int i = 0; i < shape[level]; i++) {
                    int cache_level = cache.shape.length - shape.length + level;
                    if(cache_level < 0 || cache.shape[cache_level] == 1)
                        traverse_item(level + 1, start * shape[level] + i, cache_start, op);
                    else
                        traverse_item(level + 1, start * shape[level] + i, cache_start * shape[level] + i, op);
                }
            }
        }
        private tensor calculate_item(tensor b, char op){
            // 默认初始化的对象直接返回操作的对象
            if(data == null) return b;
            int[] rtn_shape = calculate_out_shape(b);
            tensor rtn = new tensor(rtn_shape);
            rtn.set_cache(this);
            rtn.traverse_item(0, 0, 0,'=');
            rtn.set_cache(b);
            rtn.traverse_item(0, 0, 0, op);
            rtn.empty_cache();
            return rtn;
        }

        public tensor add(tensor b){
            return calculate_item(b, '+');
        }
        public tensor multiple(tensor b){
            return calculate_item(b, '*');
        }
        public tensor divide(tensor b){
            return calculate_item(b, '/');
        }
        public tensor sqrt(){
            double[] sqrt_data = new double[data.length];
            for (int i = 0; i < data.length; i++) {
                sqrt_data[i] = Math.sqrt(data[i]);
            }
            return new tensor(shape, sqrt_data);
        }
        @Override
        public String toString(){
            if(data == null) return "null";
            StringBuilder sb = new StringBuilder();
            sb.append("[".repeat(shape.length));
            int num = 1, base = 1;
            for(int sp: shape) num *= sp;
            for (int i = 1; i <= data.length; i++) {
                sb.append(data[i - 1]);
                base = 1;
                for (int k : shape) {
                    if ((i % (num / base)) == 0) sb.append(']');
                    base *= k;
                }
                if (i != data.length){
                    sb.append(", ");
                    base = 1;
                    for (int k : shape) {
                        if ((i % (num / base)) == 0) sb.append('[');
                        base *= k;
                    }
                }
            }
            return sb.toString();
        }

    }
    public static tensor calculate(tensor a, tensor b, int dim){
        // 判断维度是否匹配
        assert a.getDimension(dim) == b.getDimension(dim): "维度不匹配";
        List<tensor> a_tensors = a.getTensors(dim);
        List<tensor> b_tensors = b.getTensors(dim);
        tensor x = new tensor();
        tensor y1 = new tensor();
        tensor y2 = new tensor();

        for (int i = 0; i < a.getDimension(dim); i++) {
            x = x.add(a_tensors.get(i).multiple(b_tensors.get(i)));
            y1 = y1.add(a_tensors.get(i).multiple(a_tensors.get(i)));
            y2 = y2.add(b_tensors.get(i).multiple(b_tensors.get(i)));
        }

        return x.divide(y1.sqrt().multiple(y2.sqrt()));
    }

    public static tensor calculate(tensor a, tensor b){
        return calculate(a, b, 0);
    }

    // 测试同维，三维
    public static void test_three_dim(){
        int[] shape_a = new int[]{1, 2, 3};
        double[] data_a = new double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        int[] shape_b = new int[]{1, 1, 3};
        double[] data_b = new double[]{1.0, 4.0, 6.0};
        tensor a = new tensor(shape_a, data_a);
        tensor b = new tensor(shape_b, data_b);
        tensor result = calculate(a, b, -1);
        System.out.println(result);
    }

    // 测试同维，两维
    public static void test_two_dim(){
        int[] shape_a = new int[]{3, 2};
        double[] data_a = new double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        int[] shape_b = new int[]{3, 1};
        double[] data_b = new double[]{1.0, 4.0, 6.0};
        tensor a = new tensor(shape_a, data_a);
        tensor b = new tensor(shape_b, data_b);
        tensor result = calculate(a, b, 0);
        System.out.println(result);
    }
    // 测试同维，一维
    public static void test_one_dim(){
        int[] shape_a = new int[]{3};
        double[] data_a = new double[]{1.0, 2.0, 3.0};
        int[] shape_b = new int[]{3};
        double[] data_b = new double[]{1.0, 4.0, 6.0};
        tensor a = new tensor(shape_a, data_a);
        tensor b = new tensor(shape_b, data_b);
        tensor result = calculate(a, b);
        System.out.println(result);
    }
    // 测试不同维
    public static void test_different_dim(){
        int[] shape_a = new int[]{1, 3, 2};
        double[] data_a = new double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        int[] shape_b = new int[]{3, 1};
        double[] data_b = new double[]{1.0, 4.0, 6.0};
        tensor a = new tensor(shape_a, data_a);
        tensor b = new tensor(shape_b, data_b);
        tensor result = calculate(a, b, -2);
        System.out.println(result);
    }
    // 测试不同维,但是输入计算维度为最后一维
    public static void test_different_dim_end(){
        int[] shape_a = new int[]{1, 2, 3};
        double[] data_a = new double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        int[] shape_b = new int[]{1, 3};
        double[] data_b = new double[]{1.0, 4.0, 6.0};
        tensor a = new tensor(shape_a, data_a);
        tensor b = new tensor(shape_b, data_b);
        tensor result = calculate(a, b, -1);
        System.out.println(result);
    }
    // 测试除零，返回0
    public static void test_divide_0(){
        int[] shape_a = new int[]{1, 2, 3};
        int[] shape_b = new int[]{1, 3};
        double[] data_b = new double[]{1.0, 4.0, 6.0};
        tensor a = new tensor(shape_a);
        tensor b = new tensor(shape_b, data_b);
        tensor result = calculate(a, b, -1);
        System.out.println(result);
    }
    public static void main(String[] args){
        // 一维
        test_one_dim();
        // 两维
        test_two_dim();
        // 三维
        test_three_dim();
        // 除0
        test_divide_0();
        // 不同维
        test_different_dim();
        // 不同维，计算最后一维
        test_different_dim_end();
    }
}
