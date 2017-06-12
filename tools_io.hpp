#pragma once

#include <string>
#include <fstream>
#include <iostream>

namespace Tools {
namespace IO {
  namespace {
    /**
     * filestream object for output stream
     */
    std::ofstream ofs;
    /**
     * filestream object for error stream
     */
    std::ofstream efs;
    /**
     * filestream object for input stream
     */
    std::ifstream ifs;
    /**
     * meta function to open/close file-stream
     */
    template <typename FILE_STREAM>
    void
    set_stream(std::string fname, FILE_STREAM fh) {
      if (fh.is_open()) {
        fh.close();
      }
      if (fname != "") {
        fh.open(fname);
      }
    }
    /**
     * meta function to retrieve file-/console-stream
     * reference depending on stream status.
     */
    template <typename FILE_STREAM, typename STREAM>
    STREAM
    get_stream(FILE_STREAM fh, STREAM console) {
      if (fh.is_open()) {
        return fh;
      } else {
        return console;
      }
    }
  } // end local namespace

  /**
   * reference to out-stream.
   * either file (if set) or std::cout.
   */
  std::ostream&
  out();
  /**
   * reference to err-stream.
   * either file (if set) or std::cerr.
   */
  std::ostream&
  err();
  /**
   * reference to in-stream.
   * either file (if set) or std::cin.
   */
  std::istream&
  in();
  /**
   * set out-stream to given file.
   * if fname=="": set to std::cout.
   */
  void
  set_out(std::string fname);
  /**
   * set err-stream to given file.
   * if fname=="": set to std::cerr.
   */
  void
  set_err(std::string fname);
  /**
   * set in-stream to given file.
   * if fname=="": set to std::cin.
   */
  void
  set_in(std::string fname);
} // end namespace Tools::IO
} // end namespace Tools

