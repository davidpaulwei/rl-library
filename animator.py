import matplotlib.pyplot as plt

class animator:
    def __init__(self, xlabel: str = None, ylabel: str = None) -> None:
        self.X = []
        self.Y = []
        self.xlabel = xlabel
        self.ylabel = ylabel
    
    def add(self, x, y, text=False):
        self.X.append(x)
        self.Y.append(y)
        plt.clf()
        l1, = plt.plot(self.X, self.Y)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.legend(handles=[l1], labels=[self.ylabel], loc='best')
        if text:
            for x, y in zip(self.X, self.Y):
                plt.text(x, y, '%.2f' % y, ha = 'center', va = 'bottom')
        plt.pause(0.05)

    def stay(self):
        plt.show()

class animator2:
    def __init__(self, xlabel=None, y1label=None, y2label=None, text=False):
        self.X = []
        self.Y1 = []
        self.Y2 = []
        self.xlabel = xlabel
        self.y1label = y1label
        self.y2label = y2label
        self.text = text

    def add(self, x, y1, y2):
        self.X.append(x)
        self.Y1.append(y1)
        self.Y2.append(y2)
        plt.clf()
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax1.set_xlabel(self.xlabel)
        ax1.set_ylabel(self.y1label, color='b')
        ax2.set_ylabel(self.y2label, color='r')
        l1, = ax1.plot(self.X, self.Y1, color='b')
        l2, = ax2.plot(self.X, self.Y2, color='r')
        plt.legend(handles=[l1, l2], labels=[self.y1label, self.y2label], loc='best')
        if self.text:
            for x, y in zip(self.X, self.Y1):
                ax1.text(x, y, '%.4f' % y, ha = 'center', va = 'bottom', color='b')
            for x, y in zip(self.X, self.Y2):
                ax2.text(x, y, '%.4f' % y, ha = 'center', va = 'bottom', color='r')
        plt.pause(0.05)
    
    def stay(self):
        plt.show()

class animator3:
    def __init__(self, xlabel=None, ylabel=None, plot_num_on_xaxis=1, plot_num_on_yaxis=1) -> None:
        self.X = []
        self.Y = []
        for plot in range(plot_num_on_xaxis * plot_num_on_yaxis):
            self.X.append([])
            self.Y.append([])
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.plot_num_on_xaxis = plot_num_on_xaxis
        self.plot_num_on_yaxis = plot_num_on_yaxis
    
    def add(self, X, Y):
        plt.clf()
        for plot in range(self.plot_num_on_xaxis * self.plot_num_on_yaxis):
            self.X[plot].append(X[plot])
            self.Y[plot].append(Y[plot])
            plt.subplot(self.plot_num_on_xaxis, self.plot_num_on_yaxis, plot + 1)
            l1, = plt.plot(self.X[plot], self.Y[plot])
            plt.xlabel(self.xlabel)
            plt.ylabel(self.ylabel)
        plt.pause(0.05)

    def stay(self):
        plt.show()

# class animator4:

#     def __init__():