class HydradReader:
    def __init__(self, hydrad_dir, strand_num=0):
        """
        Initialize HydradReader with HYDRAD output directory and strand number.
        
        Parameters:
        -----------
        hydrad_dir : str
            Path to HYDRAD output directory
        strand_num : int, optional
            Index of strand to read (default: 0)
        """
        self.hydrad_dir = hydrad_dir
        self.strand_num = strand_num

    def read_strand(self):
        """
        Read the specified strand from HYDRAD output.
        
        Returns:
        --------
        pydrad.parse.parse.Strand
            The requested strand object
        """
        from pydrad.parse.parse import Strand
        return Strand(self.hydrad_dir)[self.strand_num]
